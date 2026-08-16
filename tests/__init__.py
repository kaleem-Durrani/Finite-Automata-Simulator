"""Test support package.

Exists so ``from tests import oracle`` resolves for the type checker as well as
at runtime -- the differential suite imports the oracle bridge as a module, and
without this mypy cannot place it.
"""
