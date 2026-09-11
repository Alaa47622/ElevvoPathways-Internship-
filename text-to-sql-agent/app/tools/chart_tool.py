from typing import Any
import matplotlib.pyplot as plt
import pandas as pd

def should_chart(question: str, columns: list[str], rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2 or len(rows) > 30 or len(columns) < 2:
        return False
    q = question.lower()
    return any(k in q for k in ("top", "trend", "over time", "by ", "per ", "compare", "revenue"))

def make_chart(rows, columns, question):
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in columns if c not in numeric]
    if not numeric or not categorical:
        return None
    x, y = categorical[0], numeric[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(df) <= 12:
        ax.bar(df[x].astype(str), df[y])
    else:
        ax.plot(df[x].astype(str), df[y], marker="o")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(question)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return fig
