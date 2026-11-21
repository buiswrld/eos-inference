"""Amortised multi-messenger neutron-star EOS inference (revision v2).

Lightweight on import: torch/scipy are only imported by the modules that need them,
so ``nseos.simulator`` and ``nseos.likelihood`` can be unit-tested with numpy alone.
"""
__version__ = "2.0.0"
