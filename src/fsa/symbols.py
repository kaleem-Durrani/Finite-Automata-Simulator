"""State identifiers and alphabet symbols."""

from typing import FrozenSet, Iterable

from fsa.errors import IllegalSymbolError

# A state's identity. Opaque to the engine: it never parses, orders or derives
# meaning from these, so a front end can use whatever scheme it likes and rely
# on the id surviving every operation unchanged.
StateId = str

# One character of the input alphabet.
Symbol = str


def is_legal_symbol(value: object) -> bool:
    """
    Whether ``value`` can be an alphabet symbol.

    A symbol is exactly one printable, non-whitespace character. Multi-character
    symbols are excluded because a word is simulated by iterating its
    characters, so a two-character symbol could never be read.
    """
    return (
        isinstance(value, str)
        and len(value) == 1
        and value.isprintable()
        and not value.isspace()
    )


def check_symbol(value: object) -> Symbol:
    """Return ``value`` as a symbol, or raise :class:`IllegalSymbolError`."""
    if not is_legal_symbol(value):
        raise IllegalSymbolError(f"not a valid symbol: {value!r}")
    return str(value)


def normalize_alphabet(symbols: Iterable[object]) -> FrozenSet[Symbol]:
    """Validate an iterable of symbols and return it as a frozenset."""
    return frozenset(check_symbol(symbol) for symbol in symbols)
