"""Small display formatting helpers."""
def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def money(value: float) -> str:
    return f"${value:,.2f}"


def compact_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return money(value)
