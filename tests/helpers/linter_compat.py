import re


def _norm(s: str) -> str:
    """Normalize whitespace, quote style, and Prettier trailing commas."""
    # Collapse all whitespace into single spaces
    s = re.sub(r"\s+", " ", s)
    # Normalize double quotes to single quotes
    s = s.replace('"', "'")
    # Strip spaces immediately inside parentheses
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    # Strip trailing commas inserted by Prettier before closing brackets/parens
    s = s.replace(",)", ")")
    s = s.replace(",]", "]")
    return s.strip()
