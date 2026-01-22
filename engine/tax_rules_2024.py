# tax_rules_2024.py

def calculate_pita_2024(annual_income: float) -> dict:
    """
    Nigeria PITA 2024 
    """

    # Consolidated Relief Allowance (CRA)
    cra = max(0.01 * annual_income, 200_000) + 0.20 * annual_income
    taxable_income = max(0, annual_income - cra)

    bands = [
        (300_000, 0.07),
        (300_000, 0.11),
        (500_000, 0.15),
        (500_000, 0.19),
        (1_600_000, 0.21),
        (float("inf"), 0.24),
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
    }