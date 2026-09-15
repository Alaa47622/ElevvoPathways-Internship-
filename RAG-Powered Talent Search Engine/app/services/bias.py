from collections import Counter
from typing import Any


def demographic_bias_report(
    candidates: list[dict[str, Any]],
    field: str = "gender",
    threshold: float = 0.8,
) -> dict[str, Any]:
    values = [
        str(candidate.get(field, "")).strip()
        for candidate in candidates
        if str(candidate.get(field, "")).strip()
    ]

    if not values:
        return {
            "demographic_field": field,
            "groups": [],
            "representation_ratio": None,
            "warning": f"No '{field}' data was available for this audit.",
        }

    counts = Counter(values)
    total = len(values)
    rates = {group: count / total for group, count in counts.items()}
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    ratio = min_rate / max_rate if max_rate else None

    warning = None
    if ratio is not None and ratio < threshold:
        warning = (
            f"Potential representation disparity detected: ratio {ratio:.2f} "
            f"is below threshold {threshold:.2f}. Investigate the data and "
            "retrieval pipeline; this is not proof of discriminatory behavior."
        )

    return {
        "demographic_field": field,
        "groups": [
            {
                "group": group,
                "count": count,
                "selection_rate": round(rates[group], 4),
            }
            for group, count in sorted(counts.items())
        ],
        "representation_ratio": round(ratio, 4) if ratio is not None else None,
        "warning": warning,
    }
