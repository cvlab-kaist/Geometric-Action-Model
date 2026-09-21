"""Portable inference for the AgiBot/OXE shared 16D GAM checkpoint family."""


def __getattr__(name):
    if name == "MobileGAMPolicy":
        from .policy import MobileGAMPolicy
        return MobileGAMPolicy
    raise AttributeError(name)


__all__ = ["MobileGAMPolicy"]
