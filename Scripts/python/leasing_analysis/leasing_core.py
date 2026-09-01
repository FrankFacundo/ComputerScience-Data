"""Company car (leasing) vs cash allowance + own car, Luxembourg.

Pure stdlib. `app.py` (Streamlit) and `report.py` both drive this module.

The comparison is done on after-tax cash flows over a horizon, plus the
terminal equity in the vehicle, discounted to today. The two options are
NOT symmetric and the asymmetries are what the model exists to price:

  * the leasing rental is paid by the employer and never touches your
    payroll - you are taxed only on a benefit in kind (BIK) that is a
    small percentage of the car's list price;
  * the cash alternative is ordinary salary and takes the full tax +
    social-security wedge (~50% in class 1);
  * only the owner carries depreciation, and only the owner keeps the
    residual value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from lux_tax import (
    TaxParams,
    cost_of_benefit_in_kind,
    net_of_extra_gross,
    tax_saving_from_deduction,
)

# ==========================================================================
# Benefit in kind
# ==========================================================================
# Luxembourg BIK = monthly percentage of the vehicle's list price when new,
# INCLUDING options and VAT, MINUS any discount granted by the seller.
#
# Rates (reform of 1 Jan 2025, EV transition extended on 29 Nov 2024):
#   combustion / hybrid, registered from 1 Jan 2025 ............ 2.0%
#   100% electric or H2 fuel cell, <= 18 kWh/100km ............. 0.5%   <- until 2026
#   100% electric or H2 fuel cell,  > 18 kWh/100km ............. 0.6%   <- until 2026
#   the same two, registered from 1 Jan 2027 ................... 1.0% / 1.2%
#
# The favourable EV rates survive for the WHOLE life of the contract if the
# car is registered by 31 Dec 2026, or by 31 Dec 2027 where the contract was
# signed before 31 Dec 2026. That deadline is a real, dated option.

BIK_EV_LOW_UNTIL_2026 = 0.005
BIK_EV_HIGH_UNTIL_2026 = 0.006
BIK_EV_LOW_FROM_2027 = 0.010
BIK_EV_HIGH_FROM_2027 = 0.012
BIK_COMBUSTION = 0.020
EV_CONSUMPTION_THRESHOLD = 18.0  # kWh / 100 km, WLTP


def bik_rate(
    is_electric: bool,
    consumption_kwh_100km: float,
    contract_signed_by_end_2026: bool = True,
) -> float:
    """Monthly BIK percentage applicable to a vehicle."""
    if not is_electric:
        return BIK_COMBUSTION
    low = consumption_kwh_100km <= EV_CONSUMPTION_THRESHOLD
    if contract_signed_by_end_2026:
        return BIK_EV_LOW_UNTIL_2026 if low else BIK_EV_HIGH_UNTIL_2026
    return BIK_EV_LOW_FROM_2027 if low else BIK_EV_HIGH_FROM_2027


@dataclass
class Car:
    """A vehicle. `list_price` drives the BIK, so it is the number to get right."""

    name: str
    list_price: float               # EUR TTC, new, options included ("valeur a neuf")
    lease_monthly: float            # EUR/month quoted by the employer's leaser
    consumption_kwh_100km: float    # WLTP
    is_electric: bool = True
    body: str = ""
    price_is_estimate: bool = True  # True until replaced by the figure on the quote

    def bik_monthly(self, contract_signed_by_end_2026: bool = True) -> float:
        return self.list_price * bik_rate(
            self.is_electric, self.consumption_kwh_100km, contract_signed_by_end_2026
        )


# ==========================================================================
# Scenario inputs
# ==========================================================================
@dataclass
class Common:
    base_gross_salary: float = 85_000.0   # annual gross, excluding the car budget
    horizon_months: int = 36
    annual_km: float = 15_000.0
    energy_price_per_kwh: float = 0.25    # home charging, LU, incl. taxes
    discount_rate_annual: float = 0.03


@dataclass
class LeasePolicy:
    """The employer's car policy."""

    budget_monthly: float = 535.0
    # what the rental covers
    includes_insurance: bool = True
    includes_maintenance: bool = True
    includes_tyres: bool = True
    includes_road_tax: bool = True
    includes_energy: bool = False         # charge card / fuel card
    # if the chosen car costs more than the budget, you top up out of net salary
    overbudget_from_net: bool = True
    # if it costs less, is the difference paid back to you as cash?
    underbudget_refunded: bool = False
    # cash alternative offered instead of the car, as GROSS salary
    cash_alternative_gross: float = 535.0
    # if you leave the company mid-contract, the share of the remaining
    # rentals you are liable for
    early_exit_penalty_share: float = 0.5
    contract_signed_by_end_2026: bool = True


@dataclass
class OwnCarPolicy:
    """Buying the car yourself with a consumer loan."""

    purchase_price: float = 55_000.0
    down_payment: float = 0.0
    loan_rate_apr: float = 0.055          # LU consumer car loan, TAEG
    loan_term_months: int = 60
    insurance_annual: float = 950.0
    maintenance_annual: float = 400.0
    tyres_annual: float = 300.0
    road_tax_annual: float = 30.0         # EV in LU sits at the floor
    inspection_annual: float = 25.0       # controle technique, amortised
    # share of the purchase price still recoverable at the horizon
    residual_value_pct: float = 0.50
    # a used car is cheaper to buy but the same to run
    is_electric: bool = True
    consumption_kwh_100km: float = 17.0


# ==========================================================================
# Loan mechanics
# ==========================================================================
def annuity_payment(principal: float, annual_rate: float, months: int) -> float:
    if months <= 0 or principal <= 0:
        return 0.0
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return principal / months
    return principal * r / (1.0 - (1.0 + r) ** (-months))


def amortisation(principal: float, annual_rate: float, months: int) -> List[Dict]:
    """Month-by-month schedule: payment, interest, principal, closing balance."""
    pay = annuity_payment(principal, annual_rate, months)
    r = annual_rate / 12.0
    balance = principal
    rows = []
    for m in range(1, months + 1):
        interest = balance * r
        capital = min(pay - interest, balance)
        balance = max(0.0, balance - capital)
        rows.append(
            {
                "month": m,
                "payment": pay,
                "interest": interest,
                "principal": capital,
                "balance": balance,
            }
        )
    return rows


def affordable_price(
    monthly_payment: float,
    annual_rate: float,
    months: int,
    down_payment: float = 0.0,
) -> float:
    """Inverse of the annuity: what a given monthly payment buys."""
    r = annual_rate / 12.0
    if months <= 0:
        return down_payment
    if abs(r) < 1e-12:
        return down_payment + monthly_payment * months
    return down_payment + monthly_payment * (1.0 - (1.0 + r) ** (-months)) / r


# ==========================================================================
# Scenario A - company car
# ==========================================================================
def energy_cost_annual(km: float, kwh_100: float, price_kwh: float) -> float:
    return km / 100.0 * kwh_100 * price_kwh


def evaluate_company_car(
    car: Car, common: Common, policy: LeasePolicy, tax: TaxParams
) -> Dict:
    n = common.horizon_months
    years = n / 12.0

    bik_m = car.bik_monthly(policy.contract_signed_by_end_2026)
    bik_annual = bik_m * 12.0
    bik = cost_of_benefit_in_kind(common.base_gross_salary, bik_annual, tax)
    bik_cost_monthly = bik["cost"] / 12.0

    # top-up if the car is above the policy budget
    overbudget = max(0.0, car.lease_monthly - policy.budget_monthly)
    if overbudget and not policy.overbudget_from_net:
        # taken from gross salary: costs you only the net equivalent
        overbudget = net_of_extra_gross(
            common.base_gross_salary, overbudget * 12.0, tax
        )["net"] / 12.0

    # budget left on the table if you pick a cheaper car
    underbudget = max(0.0, policy.budget_monthly - car.lease_monthly)
    refund_monthly = 0.0
    if underbudget and policy.underbudget_refunded:
        refund_monthly = (
            net_of_extra_gross(common.base_gross_salary, underbudget * 12.0, tax)["net"]
            / 12.0
        )

    energy_monthly = 0.0
    if not policy.includes_energy:
        energy_monthly = (
            energy_cost_annual(
                common.annual_km, car.consumption_kwh_100km, common.energy_price_per_kwh
            )
            / 12.0
        )

    monthly_out = bik_cost_monthly + overbudget + energy_monthly - refund_monthly
    flows = [monthly_out] * n

    return {
        "label": f"Company car - {car.name}",
        "bik_rate": bik_rate(
            car.is_electric, car.consumption_kwh_100km, policy.contract_signed_by_end_2026
        ),
        "bik_monthly": bik_m,
        "bik_tax_cost_monthly": bik_cost_monthly,
        "overbudget_monthly": overbudget,
        "refund_monthly": refund_monthly,
        "energy_monthly": energy_monthly,
        "monthly_out_of_pocket": monthly_out,
        "flows": flows,
        "employer_spend": car.lease_monthly * n,
        "total_nominal_cost": sum(flows),
        "terminal_equity": 0.0,
        "net_cost_nominal": sum(flows),
        "npv": npv(flows, common.discount_rate_annual, terminal=0.0),
        "years": years,
    }


# ==========================================================================
# Scenario B - cash allowance + own car
# ==========================================================================
def evaluate_own_car(
    common: Common, policy: LeasePolicy, own: OwnCarPolicy, tax: TaxParams
) -> Dict:
    n = common.horizon_months

    # 1. the cash you actually keep from the allowance
    cash = net_of_extra_gross(
        common.base_gross_salary, policy.cash_alternative_gross * 12.0, tax
    )
    cash_net_monthly = cash["net"] / 12.0

    # 2. the loan
    principal = max(0.0, own.purchase_price - own.down_payment)
    sched = amortisation(principal, own.loan_rate_apr, own.loan_term_months)
    pay = sched[0]["payment"] if sched else 0.0

    # 3. running costs the leasing would otherwise have covered
    running_annual = (
        own.insurance_annual
        + own.maintenance_annual
        + own.tyres_annual
        + own.road_tax_annual
        + own.inspection_annual
    )
    running_monthly = running_annual / 12.0
    energy_monthly = (
        energy_cost_annual(
            common.annual_km, own.consumption_kwh_100km, common.energy_price_per_kwh
        )
        / 12.0
    )

    # 4. the deduction the user hoped for - capped at the special-expenses ceiling
    interest_by_year: Dict[int, float] = {}
    for row in sched[:n]:
        y = (row["month"] - 1) // 12
        interest_by_year[y] = interest_by_year.get(y, 0.0) + row["interest"]
    interest_paid = sum(interest_by_year.values())
    deductible = {
        y: min(amt, tax.interest_headroom) for y, amt in interest_by_year.items()
    }
    tax_saving_by_year = {
        y: tax_saving_from_deduction(common.base_gross_salary, amt, tax)
        for y, amt in deductible.items()
    }
    total_tax_saving = sum(tax_saving_by_year.values())

    # 5. monthly flows (loan may run past, or end before, the horizon)
    flows: List[float] = []
    for m in range(1, n + 1):
        loan_m = pay if m <= own.loan_term_months else 0.0
        f = loan_m + running_monthly + energy_monthly - cash_net_monthly
        if m % 12 == 0:  # tax refunds land once a year
            f -= tax_saving_by_year.get((m - 1) // 12, 0.0)
        flows.append(f)
    if own.down_payment:
        flows[0] += own.down_payment

    # 6. terminal position: what the car is worth minus what you still owe
    resale = own.purchase_price * own.residual_value_pct
    balance = sched[n - 1]["balance"] if n <= len(sched) else 0.0
    terminal_equity = resale - balance

    return {
        "label": "Cash allowance + own car",
        "cash_gross_monthly": policy.cash_alternative_gross,
        "cash_net_monthly": cash_net_monthly,
        "keep_ratio": cash["keep_ratio"],
        "loan_principal": principal,
        "loan_payment_monthly": pay,
        "running_monthly": running_monthly,
        "energy_monthly": energy_monthly,
        "interest_paid_horizon": interest_paid,
        "interest_deductible_total": sum(deductible.values()),
        "tax_saving_total": total_tax_saving,
        "resale_value": resale,
        "loan_balance_at_horizon": balance,
        "terminal_equity": terminal_equity,
        "flows": flows,
        "total_nominal_cost": sum(flows),
        "net_cost_nominal": sum(flows) - terminal_equity,
        "npv": npv(flows, common.discount_rate_annual, terminal=terminal_equity),
        "years": n / 12.0,
    }


# ==========================================================================
# Discounting
# ==========================================================================
def npv(flows: List[float], annual_rate: float, terminal: float = 0.0) -> float:
    r = annual_rate / 12.0
    total = 0.0
    for i, f in enumerate(flows, start=1):
        total += f / ((1.0 + r) ** i)
    if terminal:
        total -= terminal / ((1.0 + r) ** len(flows))
    return total


# ==========================================================================
# Derived analyses
# ==========================================================================
def early_exit_cost(car: Car, policy: LeasePolicy, leave_after_months: int,
                    horizon_months: int) -> float:
    """Penalty if you leave the company (or the car) before term."""
    remaining = max(0, horizon_months - leave_after_months)
    return remaining * car.lease_monthly * policy.early_exit_penalty_share


def breakeven_purchase_price(
    car: Car,
    common: Common,
    policy: LeasePolicy,
    own: OwnCarPolicy,
    tax: TaxParams,
    lo: float = 1_000.0,
    hi: float = 200_000.0,
    tol: float = 1.0,
) -> Optional[float]:
    """Price of a self-bought car whose NPV equals the company car's.

    Below this price, buying wins; above it, the company car wins.
    """
    target = evaluate_company_car(car, common, policy, tax)["npv"]

    def f(price: float) -> float:
        o = replace(own, purchase_price=price)
        return evaluate_own_car(common, policy, o, tax)["npv"] - target

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        return None  # even the cheapest car cannot beat the company car
    if f_hi < 0:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def budget_to_car_price(
    monthly_budget: float,
    own: OwnCarPolicy,
    common: Common,
    include_running: bool = True,
) -> float:
    """What list price a given all-in monthly budget supports when buying.

    Running costs are carved out of the budget first, because the loan is not
    the only bill an owner pays.
    """
    running = 0.0
    if include_running:
        running = (
            own.insurance_annual
            + own.maintenance_annual
            + own.tyres_annual
            + own.road_tax_annual
            + own.inspection_annual
        ) / 12.0
    available = max(0.0, monthly_budget - running)
    return affordable_price(
        available, own.loan_rate_apr, own.loan_term_months, own.down_payment
    )


# ==========================================================================
# The cars quoted by BNP
# ==========================================================================
# list_price = estimated Luxembourg "valeur a neuf" TTC with the quoted trim.
# THESE ARE ESTIMATES. The BIK is a straight percentage of this number, so
# replace each one with the figure written on the leaser's quote before you
# rely on the euros below.
BNP_CARS: List[Car] = [
    Car("BMW iX2 eDrive20 M Edition",        52_500, 486, 16.5, True, "SUV"),
    Car("BMW iX1 eDrive20 M Edition",        51_500, 489, 16.8, True, "SUV"),
    Car("Mercedes CLA 250+ Business Line",   56_000, 548, 14.5, True, "Berline"),
    Car("Mercedes CLA SB 250+ Business",     57_500, 561, 15.0, True, "Break"),
    Car("BMW i4 eDrive40 M Edition",         61_000, 555, 16.5, True, "Berline"),
    Car("Mercedes GLB 250+ Business Line",   60_000, 571, 18.5, True, "SUV 7pl"),
    Car("Mercedes GLC 250 Business Line",    68_000, 657, 18.5, True, "SUV"),
    Car("Mercedes C 400 4MATIC Business",    72_000, 667, 19.5, True, "Berline"),
]


__all__ = [
    "Car", "Common", "LeasePolicy", "OwnCarPolicy", "BNP_CARS",
    "bik_rate", "annuity_payment", "amortisation", "affordable_price",
    "evaluate_company_car", "evaluate_own_car", "npv", "early_exit_cost",
    "breakeven_purchase_price", "budget_to_car_price", "energy_cost_annual",
    "BIK_EV_LOW_UNTIL_2026", "BIK_EV_HIGH_UNTIL_2026", "BIK_COMBUSTION",
    "EV_CONSUMPTION_THRESHOLD", "replace",
]
