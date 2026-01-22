# tax_rules_2026.py

def calculate_pita_2026(annual_income: float) -> dict:
    """
    Proposed reform – lower burden on lower income
    """

    cra = max(0.02 * annual_income, 250_000) + 0.25 * annual_income
    taxable_income = max(0, annual_income - cra)

    bands = [
        (500_000, 0.05),
        (500_000, 0.10),
        (1_000_000, 0.15),
        (2_000_000, 0.20),
        (float("inf"), 0.22),
    ]

    tax = 0
    remaining = taxable_income

    for limit, rate in bands:
        if remaining <= 0:
            break
        applied = min(remaining, limit)
        tax += applied * rate
        remaining -= applied

    effective_rate = (tax / annual_income) * 100 if annual_income else 0

    return {
        "annual_tax": round(tax, 2),
        "effective_rate": round(effective_rate, 2),
        "taxable_income": round(taxable_income, 2),
        "classification": "pro-poor" if annual_income < 10_000_000 else "neutral",
    }
