"""Luxembourg payroll tax / social-security engine (resident employee).

Pure stdlib so it runs anywhere. Every rate is a dataclass field so the
Streamlit front-end can override it without touching this module.

Sources for the defaults are listed in REPORT.md. Figures are the
2025/2026 scale; the income-tax bracket table was NOT re-indexed for 2026
(Luxembourg only adjusts it every three index tranches, starting 2028),
so the same table serves both years.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple

# --------------------------------------------------------------------------
# Income-tax scale, tax class 1, 2025 (indexed). Each entry is
# (lower bound of the band in EUR of annual taxable income, marginal rate).
# --------------------------------------------------------------------------
SCALE_2025_CLASS1: List[Tuple[float, float]] = [
    (0, 0.00),
    (13_200, 0.08),
    (15_400, 0.09),
    (17_600, 0.10),
    (19_800, 0.11),
    (22_050, 0.12),
    (24_250, 0.14),
    (26_550, 0.16),
    (28_800, 0.18),
    (31_100, 0.20),
    (33_400, 0.22),
    (35_700, 0.24),
    (38_000, 0.26),
    (40_300, 0.28),
    (42_600, 0.30),
    (44_900, 0.32),
    (47_200, 0.34),
    (49_500, 0.36),
    (51_750, 0.38),
    (54_050, 0.39),
    (117_450, 0.40),
    (176_150, 0.41),
    (234_850, 0.42),
]


@dataclass
class TaxParams:
    """All Luxembourg fiscal + social parameters for a resident employee."""

    # --- income tax ---
    scale: List[Tuple[float, float]] = field(
        default_factory=lambda: list(SCALE_2025_CLASS1)
    )
    tax_class: str = "1"  # "1" or "2" (joint / splitting)
    scale_index_factor: float = 1.0  # uprate the whole scale if it gets indexed

    # solidarity surcharge (contribution au fonds pour l'emploi)
    solidarity_rate: float = 0.07
    solidarity_rate_high: float = 0.09
    solidarity_threshold: float = 150_000.0  # on adjusted taxable income

    # --- employee social security (2026 rates) ---
    pension_rate: float = 0.085          # 8.5% from 2026 (was 8.0%)
    sickness_rate: float = 0.0305        # 2.80% in-kind + 0.25% cash benefits
    dependency_rate: float = 0.014       # assurance dependance, no ceiling
    socsec_ceiling_monthly: float = 13_856.63
    dependency_abatement_monthly: float = 692.83

    # --- standard deductions that shift the bracket position ---
    frais_obtention: float = 540.0       # minimum lump-sum professional expenses
    depenses_speciales_min: float = 480.0

    # --- special-expenses ceiling for debt interest + insurance premiums ---
    # This is the ceiling the car-loan interest has to fit inside.
    interest_premium_ceiling: float = 672.0   # per person, per year
    interest_premium_ceiling_used: float = 0.0  # already consumed by other premiums

    @property
    def socsec_ceiling_annual(self) -> float:
        return self.socsec_ceiling_monthly * 12.0

    @property
    def dependency_abatement_annual(self) -> float:
        return self.dependency_abatement_monthly * 12.0

    @property
    def interest_headroom(self) -> float:
        """Debt interest that can still actually be deducted, per year."""
        return max(0.0, self.interest_premium_ceiling - self.interest_premium_ceiling_used)


# --------------------------------------------------------------------------
# Core primitives
# --------------------------------------------------------------------------
def _scale_tax(taxable: float, scale: List[Tuple[float, float]], factor: float) -> float:
    """Progressive tax from a band table, applied to a single taxable income."""
    if taxable <= 0:
        return 0.0
    tax = 0.0
    bands = [(lo * factor, rate) for lo, rate in scale]
    for i, (lo, rate) in enumerate(bands):
        hi = bands[i + 1][0] if i + 1 < len(bands) else float("inf")
        if taxable <= lo:
            break
        tax += rate * (min(taxable, hi) - lo)
    return tax


def income_tax(taxable: float, p: TaxParams) -> float:
    """Income tax before the solidarity surcharge, for the configured class."""
    if p.tax_class == "2":
        # Joint taxation: splitting. Tax = 2 x scale(income / 2).
        return 2.0 * _scale_tax(taxable / 2.0, p.scale, p.scale_index_factor)
    return _scale_tax(taxable, p.scale, p.scale_index_factor)


def social_contributions(gross_annual: float, p: TaxParams) -> dict:
    """Employee-side social security on an annual gross salary."""
    capped = min(gross_annual, p.socsec_ceiling_annual)
    pension = p.pension_rate * capped
    sickness = p.sickness_rate * capped
    dependency = p.dependency_rate * max(0.0, gross_annual - p.dependency_abatement_annual)
    return {
        "pension": pension,
        "sickness": sickness,
        "dependency": dependency,
        # dependency is NOT deductible from taxable income; the other two are
        "deductible": pension + sickness,
        "total": pension + sickness + dependency,
    }


def solidarity_surcharge(taxable: float, tax: float, p: TaxParams) -> float:
    """Contribution au fonds pour l'emploi.

    7% of the tax, rising to 9% "au-dela d'un revenu imposable ajuste de
    150 000 EUR" (300 000 in class 2). We read "au-dela" as graduated: the 9%
    applies only to the tax attributable to the income above the threshold.
    A cliff reading would make net income *fall* as gross income rises through
    the threshold, which no tax system intends. Far above the salary range
    this analysis cares about either way.
    """
    threshold = p.solidarity_threshold * (2.0 if p.tax_class == "2" else 1.0)
    if taxable <= threshold:
        return tax * p.solidarity_rate
    tax_below = income_tax(threshold, p)
    return tax_below * p.solidarity_rate + (tax - tax_below) * p.solidarity_rate_high


def net_annual(gross_annual: float, p: TaxParams, extra_deductions: float = 0.0) -> dict:
    """Full gross -> net breakdown for one year.

    `extra_deductions` are additional special expenses (e.g. deductible car-loan
    interest) that reduce taxable income.
    """
    ss = social_contributions(gross_annual, p)
    taxable = max(
        0.0,
        gross_annual
        - ss["deductible"]
        - p.frais_obtention
        - max(p.depenses_speciales_min, extra_deductions),
    )
    tax = income_tax(taxable, p)
    surcharge = solidarity_surcharge(taxable, tax, p)
    total_tax = tax + surcharge
    return {
        "gross": gross_annual,
        "social": ss["total"],
        "social_detail": ss,
        "taxable": taxable,
        "income_tax": tax,
        "solidarity": surcharge,
        "total_tax": total_tax,
        "net": gross_annual - ss["total"] - total_tax,
    }


# --------------------------------------------------------------------------
# The two numbers the whole analysis hangs on
# --------------------------------------------------------------------------
def net_of_extra_gross(base_gross: float, extra_gross: float, p: TaxParams) -> dict:
    """What you actually keep from `extra_gross` stacked on top of `base_gross`.

    Computed by differencing two full payroll runs, so bracket crossings and
    the social-security ceiling are handled exactly rather than with a
    single assumed marginal rate.
    """
    a = net_annual(base_gross, p)
    b = net_annual(base_gross + extra_gross, p)
    delta_net = b["net"] - a["net"]
    return {
        "gross": extra_gross,
        "social": b["social"] - a["social"],
        "tax": b["total_tax"] - a["total_tax"],
        "net": delta_net,
        "keep_ratio": delta_net / extra_gross if extra_gross else 0.0,
        "wedge_ratio": 1.0 - (delta_net / extra_gross if extra_gross else 0.0),
    }


def cost_of_benefit_in_kind(base_gross: float, bik_annual: float, p: TaxParams) -> dict:
    """Cash cost of a benefit in kind: you are taxed on it but receive no money.

    In Luxembourg the private use of a company car is taxable salary AND part
    of the social-security base, so the cost is the full wedge on the BIK.
    """
    w = net_of_extra_gross(base_gross, bik_annual, p)
    return {
        "bik": bik_annual,
        "social": w["social"],
        "tax": w["tax"],
        "cost": w["social"] + w["tax"],
        "cost_ratio": w["wedge_ratio"],
    }


def tax_saving_from_deduction(base_gross: float, deduction: float, p: TaxParams) -> float:
    """Cash value of an extra special-expenses deduction (already capped)."""
    if deduction <= 0:
        return 0.0
    a = net_annual(base_gross, p)
    b = net_annual(base_gross, p, extra_deductions=p.depenses_speciales_min + deduction)
    return b["net"] - a["net"]


def marginal_rate(base_gross: float, p: TaxParams, step: float = 100.0) -> float:
    """Marginal income-tax rate (incl. solidarity) at a given salary."""
    a = net_annual(base_gross, p)
    b = net_annual(base_gross + step, p)
    return (b["total_tax"] - a["total_tax"]) / step


__all__ = [
    "SCALE_2025_CLASS1",
    "TaxParams",
    "income_tax",
    "social_contributions",
    "solidarity_surcharge",
    "net_annual",
    "net_of_extra_gross",
    "cost_of_benefit_in_kind",
    "tax_saving_from_deduction",
    "marginal_rate",
    "replace",
]
