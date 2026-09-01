"""Sanity checks on the financial maths. Run: python3 test_model.py"""

from dataclasses import replace

from leasing_core import (
    BNP_CARS, Common, LeasePolicy, OwnCarPolicy, Car, affordable_price,
    amortisation, annuity_payment, bik_rate, evaluate_company_car,
    evaluate_own_car, npv,
)
from lux_tax import TaxParams, net_annual, net_of_extra_gross, social_contributions

ok, fail = 0, 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("Loan mechanics")
p = annuity_payment(10_000, 0.06, 12)
check("annuity repays exactly", abs(amortisation(10_000, 0.06, 12)[-1]["balance"]) < 1e-6)
check("interest + principal == payment",
      all(abs(r["interest"] + r["principal"] - r["payment"]) < 1e-6
          for r in amortisation(10_000, 0.06, 12)[:-1]))
check("zero-rate annuity is principal/n", abs(annuity_payment(1200, 0.0, 12) - 100) < 1e-9)
check("affordable_price inverts annuity",
      abs(affordable_price(p, 0.06, 12) - 10_000) < 1e-6)

print("\nTax engine")
t = TaxParams()
check("zero income -> zero tax", net_annual(0, t)["total_tax"] == 0)
check("net < gross", net_annual(85_000, t)["net"] < 85_000)
check("net is monotonic in gross (fine grid, both classes)",
      all(net_annual(g, tc)["net"] < net_annual(g + 250, tc)["net"]
          for tc in (t, replace(t, tax_class="2"))
          for g in range(10_000, 400_000, 250)))
check("no solidarity cliff at the class-1 threshold",
      net_annual(170_000, t)["net"] < net_annual(171_000, t)["net"])
check("class 2 solidarity threshold is doubled",
      net_annual(320_000, replace(t, tax_class="2"))["net"]
      < net_annual(321_000, replace(t, tax_class="2"))["net"])
ss = social_contributions(300_000, t)
check("pension capped at ceiling",
      abs(ss["pension"] - t.pension_rate * t.socsec_ceiling_annual) < 1e-6)
check("dependency uncapped",
      ss["dependency"] > t.dependency_rate * t.socsec_ceiling_annual)
check("class 2 keeps more than class 1",
      net_of_extra_gross(85_000, 6420, replace(t, tax_class="2"))["net"]
      > net_of_extra_gross(85_000, 6420, t)["net"])
w = net_of_extra_gross(85_000, 6420, t)
check("keep ratio in (0,1)", 0 < w["keep_ratio"] < 1, f"{w['keep_ratio']:.3f}")

print("\nBenefit in kind")
check("EV low consumption 2026 = 0.5%", bik_rate(True, 16.0, True) == 0.005)
check("EV high consumption 2026 = 0.6%", bik_rate(True, 19.0, True) == 0.006)
check("EV low consumption 2027 = 1.0%", bik_rate(True, 16.0, False) == 0.010)
check("EV threshold is inclusive at 18", bik_rate(True, 18.0, True) == 0.005)
check("combustion = 2% regardless", bik_rate(False, 5.0, True) == 0.02
      and bik_rate(False, 5.0, False) == 0.02)

print("\nScenarios")
c, pol, car = Common(), LeasePolicy(), BNP_CARS[0]
own = OwnCarPolicy(purchase_price=car.list_price)
A = evaluate_company_car(car, c, pol, t)
B = evaluate_own_car(c, pol, own, t)
check("company car cheaper for the same car", A["net_cost_nominal"] < B["net_cost_nominal"])
check("BIK tax cost is a fraction of the BIK",
      0 < A["bik_tax_cost_monthly"] < A["bik_monthly"])
check("2027 contract costs more than 2026",
      evaluate_company_car(car, c, replace(pol, contract_signed_by_end_2026=False), t)
      ["net_cost_nominal"] > A["net_cost_nominal"])
check("deductible interest respects the ceiling",
      B["interest_deductible_total"] <= t.interest_headroom * (c.horizon_months / 12) + 1e-6,
      f"{B['interest_deductible_total']:.0f}")
check("used ceiling kills the deduction",
      evaluate_own_car(c, pol, own, replace(t, interest_premium_ceiling_used=672.0))
      ["tax_saving_total"] == 0)
check("terminal equity = resale - balance",
      abs(B["terminal_equity"] - (B["resale_value"] - B["loan_balance_at_horizon"])) < 1e-6)
check("cheaper car narrows the gap",
      evaluate_own_car(c, pol, replace(own, purchase_price=15_000), t)["net_cost_nominal"]
      < B["net_cost_nominal"])
check("over-budget car costs more than in-budget",
      evaluate_company_car(BNP_CARS[-1], c, pol, t)["monthly_out_of_pocket"]
      > A["monthly_out_of_pocket"])

print("\nDiscounting")
check("zero rate NPV == sum of flows", abs(npv([100] * 12, 0.0) - 1200) < 1e-9)
check("positive rate discounts", npv([100] * 12, 0.05) < 1200)
check("terminal value reduces NPV", npv([100] * 12, 0.05, terminal=500) < npv([100] * 12, 0.05))

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
