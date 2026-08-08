"""Entry point for ``python -m fsa``.

Lets the CLI run straight from a checkout, without an install step.
"""

from fsa.cli import run_main

if __name__ == "__main__":
    run_main()
