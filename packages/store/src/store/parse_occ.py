import re

from domain.models import ParsedOccSymbol
from domain.types import is_symbol
from domain.utils import make_expiry_datetime

# Example: O:NVDA260117C00140000
# Symbol: NVDA
# Expiry: 2026-01-17
# Type: C/P
# Strike: 140.000
_OCC_PATTERN = re.compile(
    r"^O:(?P<symbol>[A-Z]+)"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<type>[CP])"
    r"(?P<strike>\d{8})$"
)


def parse_occ_symbol(occ_symbol: str) -> ParsedOccSymbol | None:
    """
    Parse OCC option symbol.

    Returns:
        ParsedOccSymbol or None if invalid.
    """
    m = _OCC_PATTERN.match(occ_symbol)
    if not m:
        return None

    symbol = m.group("symbol")
    if not is_symbol(symbol):
        return None

    expiration_date = make_expiry_datetime(f"20{m.group('yy')}-{m.group('mm')}-{m.group('dd')}")
    option_type = "call" if m.group("type") == "C" else "put"
    strike_price = int(m.group("strike")) / 1000.0

    return ParsedOccSymbol(
        symbol=symbol,
        expiration_date=expiration_date,
        option_type=option_type,
        strike_price=strike_price,
    )
