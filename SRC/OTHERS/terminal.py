from pydantic import *


def section(n, title: str) -> None:
    print("\n")
    print("=" * 78)
    print(f"  {n}.  {title}")
    print("=" * 78)


def kv(label: str, value: float) -> None:
    print(f"    {label:<34}{value}")


def table(
    headers: List[str],
    rows: List[Tuple[str, float, str]],  # ← Each row is (str, float, str)
    indent: str = "    ",
) -> None:
    """Print a formatted table with headers and rows."""
    cols = list(zip(*([headers] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    print(indent + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(headers)))
    print(indent + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print(indent + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(r)))
