"""Retired LIBERO closed-loop entry point.

The blessed entry point is now ``src/eval_libero_unified.py``. This file keeps
the old helper code importable for compatibility, but direct CLI execution is
forwarded to the unified evaluator.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from robot.modeling.da3_giant_encoder import DA3GiantEncoder
from robot.data.dataset import ActionNormalizer
from robot.evaluation.rollout_env import create_rollout_env_libero, list_libero_tasks
from train_robot import DA3FineTuneModel


LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIBERO_CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def get_obs_images(obs, camera_names, image_size):
    """Extract camera images from LIBERO obs dict, resize to model input size."""
    images = []
    for cam in camera_names:
        key = f"{cam}_image" if not cam.endswith("_image") else cam
        img = obs.get(key)
        if img is None:
            key = cam
            img = obs.get(key)
        if img is None:
            raise KeyError(f"Camera '{cam}' missing in obs keys: {sorted(obs.keys())}")
        img = np.asarray(img, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(img).resize(image_size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
    return torch.stack(images, dim=0)  # (V, 3, H, W)


def get_obs_proprio(obs):
    """Extract proprioception from LIBERO obs dict."""
    parts = []
    if "robot0_eef_pos" in obs:
        parts.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32).flatten())
    if "robot0_eef_quat" in obs:
        parts.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).flatten())
    if "robot0_gripper_qpos" in obs:
        parts.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).flatten())
    if parts:
        return torch.from_numpy(np.concatenate(parts))
    return torch.zeros(8)


def rollout_episode(
    env,
    init_state,
    model,
    normalizer,
    encoder_mean,
    encoder_std,
    device,
    image_size=(224, 224),
    max_steps=600,
    action_horizon=1,
    use_bf16=True,
    record_frames=False,
    camera_size=256,
):
    """Run one episode: model predicts actions from live observations, execute in env."""
    env.reset()
    env.base_env.sim.set_state_from_flattened(init_state)
    env.base_env.sim.forward()

    obs = env.get_observation()
    frames = []
    success = False
    step = 0

    while step < max_steps:
        # Get current observation images
        current_images = get_obs_images(obs, LIBERO_CAMERA_NAMES, image_size)  # (V, 3, H, W)
        current_proprio = get_obs_proprio(obs)

        # Model expects (B, T*V, 3, H, W) : for single-step input, T=1
        # The model needs T timesteps of input. For closed-loop, we use T=1
        # and let the model predict the action sequence.
        V = current_images.shape[0]
        images_input = current_images.unsqueeze(0).to(device)  # (1, V, 3, H, W)
        # Normalize
        images_norm = (images_input.float() - encoder_mean) / encoder_std
        proprio_input = current_proprio.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                action_pred, _ = model(images_norm, proprio=proprio_input)

        # action_pred: (1, T, 7) : take first action_horizon steps
        pred_actions = normalizer.denormalize(action_pred[0])  # (T, 7)
        pred_actions_np = pred_actions.cpu().float().numpy()

        # Execute action_horizon steps
        for ah in range(min(action_horizon, len(pred_actions_np))):
            action = pred_actions_np[ah]
            obs, reward, done, info = env.step(action)
            step += 1

            if record_frames:
                frame_panels = []
                for cam in LIBERO_CAMERA_NAMES:
                    img = env.render(mode="rgb_array", height=camera_size, width=camera_size, camera_name=cam)
                    frame_panels.append(img)
                frames.append(np.concatenate(frame_panels, axis=1))

            success = success or bool(env.is_success()["task"])
            if success or step >= max_steps:
                break

        if success:
            break

    return {
        "success": success,
        "steps": step,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO closed-loop evaluation")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results/eval_libero")
    parser.add_argument(
        "--suites", type=str, default="libero_spatial,libero_object,libero_goal,libero_10",
        help="Comma-separated LIBERO suite names",
    )
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated task IDs within suite (default: all)")
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--action-horizon", type=int, default=1, help="Actions to execute per inference")
    parser.add_argument("--camera-size", type=int, default=256, help="Render camera resolution")
    parser.add_argument("--render", action="store_true", help="Save rollout videos")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="eval-libero")
    # LIBERO-PRO perturbation options
    parser.add_argument("--pro", action="store_true", help="Enable LIBERO-PRO perturbation evaluation")
    parser.add_argument(
        "--pro-root", type=str,
        default=os.path.join(os.environ.get("DA3_ROOT", "."), "LIBERO-PRO"),
        help="Path to cloned LIBERO-PRO repository",
    )
    parser.add_argument(
        "--pro-perturbation", type=str, default="all",
        help="Perturbation type: object_attr, init_position, instruction, environment, all",
    )
    parser.add_argument(
        "--pro-displacement", type=float, default=0.1,
        help="Object displacement magnitude for init_position perturbation (default: 0.1 units)",
    )
    args = parser.parse_args()

    if args.pro:
        if not os.path.exists(args.pro_root):
            print(f"LIBERO-PRO missing at {args.pro_root}")
            print("Run: bash scripts/setup_libero_pro.sh")
            sys.exit(1)
        # Add LIBERO-PRO to path for perturbation utilities
        sys.path.insert(0, args.pro_root)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stage1_cfg = OmegaConf.to_container(cfg.get("stage_1", {}), resolve=True)
    da3_ft_cfg = OmegaConf.to_container(cfg.get("da3_finetune", {}), resolve=True)
    action_head_cfg = OmegaConf.to_container(cfg.get("action_head", {}), resolve=True)
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    proprio_cfg = OmegaConf.to_container(cfg.get("proprioception", {}), resolve=True)
    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)

    n_views = int(da3_ft_cfg.get("n_views", 2))
    future_steps = int(dataset_cfg.get("future_steps", 6))
    include_current = bool(dataset_cfg.get("include_current_action", True))
    action_steps = future_steps + 1 if include_current else future_steps
    use_bf16 = bool(training_cfg.get("bf16", True))
    image_size = tuple(dataset_cfg.get("image_size", [224, 224]))

    # --- Load model ---
    print("Loading model...")
    teacher = DA3GiantEncoder(ckpt_path=stage1_cfg["ckpt_path"])
    teacher.eval()

    model = DA3FineTuneModel(
        teacher,
        da3_finetune_cfg=da3_ft_cfg,
        action_head_cfg=action_head_cfg,
        proprio_cfg=proprio_cfg,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Strip _orig_mod. prefix from torch.compile checkpoints
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    teacher = teacher.to(device).eval()
    encoder_mean = teacher.encoder_mean.float().to(device)
    encoder_std = teacher.encoder_std.float().to(device)

    # Action normalizer : use libero stats or mimicgen stats
    normalizer = ActionNormalizer(action_dim=7)

    # --- WandB ---
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="da3-action-finetune",
            name=args.wandb_name,
            config={"eval_type": "libero", "suites": args.suites},
        )

    # --- Evaluate ---
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    all_results = {}
    summary_rows = []

    for suite_name in suites:
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name}")
        print(f"{'='*60}")

        task_names = list_libero_tasks(suite_name)
        eval_task_ids = task_ids if task_ids else list(range(len(task_names)))

        suite_successes = []

        for tid in eval_task_ids:
            task_desc = task_names[tid]
            print(f"\nTask {tid}: {task_desc}")

            try:
                env, task_name, init_states = create_rollout_env_libero(
                    suite_name=suite_name,
                    task_id=tid,
                    camera_names=LIBERO_CAMERA_NAMES,
                    camera_size=args.camera_size,
                )
            except Exception as e:
                print(f"  Failed to create env: {e}")
                continue

            n_inits = len(init_states)
            task_successes = []

            for ep in range(args.n_episodes):
                init_idx = ep % n_inits
                t0 = time.time()

                try:
                    result = rollout_episode(
                        env=env,
                        init_state=init_states[init_idx],
                        model=model,
                        normalizer=normalizer,
                        encoder_mean=encoder_mean,
                        encoder_std=encoder_std,
                        device=device,
                        image_size=image_size,
                        max_steps=args.max_steps,
                        action_horizon=args.action_horizon,
                        use_bf16=use_bf16,
                        record_frames=args.render,
                        camera_size=args.camera_size,
                    )
                    task_successes.append(result["success"])
                    elapsed = time.time() - t0
                    print(
                        f"  Episode {ep}: {'SUCCESS' if result['success'] else 'FAIL'} "
                        f"({result['steps']} steps, {elapsed:.1f}s)"
                    )

                    if args.render and result["frames"]:
                        video_dir = os.path.join(args.output_dir, suite_name)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f"task{tid}_ep{ep}.mp4")
                        try:
                            import imageio
                            imageio.mimsave(video_path, result["frames"], fps=10)
                        except Exception as e:
                            print(f"  Video save failed: {e}")

                except Exception as e:
                    print(f"  Episode {ep} failed: {e}")
                    task_successes.append(False)

            success_rate = np.mean(task_successes) if task_successes else 0.0
            suite_successes.extend(task_successes)
            print(f"  Task {tid} success rate: {success_rate:.1%} ({sum(task_successes)}/{len(task_successes)})")

            summary_rows.append({
                "suite": suite_name,
                "task_id": tid,
                "task_name": task_desc,
                "n_episodes": len(task_successes),
                "n_success": sum(task_successes),
                "success_rate": round(success_rate, 4),
            })

            if wandb_run:
                wandb.log({
                    f"libero/{suite_name}/task_{tid}_success_rate": success_rate,
                    f"libero/{suite_name}/task_{tid}_name": task_desc,
                })

            env.close()

        suite_sr = np.mean(suite_successes) if suite_successes else 0.0
        all_results[suite_name] = {
            "success_rate": float(suite_sr),
            "n_episodes": len(suite_successes),
            "n_success": int(sum(suite_successes)),
        }
        print(f"\n{suite_name} overall: {suite_sr:.1%} ({sum(suite_successes)}/{len(suite_successes)})")

        if wandb_run:
            wandb.log({f"libero/{suite_name}/success_rate": suite_sr})

    # --- Summary ---
    print(f"\n{'='*60}")
    print("LIBERO Evaluation Summary")
    print(f"{'='*60}")
    avg_sr = np.mean([v["success_rate"] for v in all_results.values()]) if all_results else 0.0
    for suite, res in all_results.items():
        print(f"  {suite}: {res['success_rate']:.1%} ({res['n_success']}/{res['n_episodes']})")
    print(f"  Average: {avg_sr:.1%}")

    if wandb_run:
        wandb.log({"libero/average_success_rate": avg_sr})

    # Save results
    results_path = os.path.join(args.output_dir, "libero_results.json")
    with open(results_path, "w") as f:
        json.dump({"suites": all_results, "average_success_rate": float(avg_sr)}, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save per-task CSV
    csv_path = os.path.join(args.output_dir, "libero_per_task.csv")
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Per-task CSV: {csv_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    print(
        "WARNING: src/eval_libero.py is retired; forwarding to src/eval_libero_unified.py",
        file=sys.stderr,
    )
    from eval_libero_unified import main as unified_main

    unified_main()
