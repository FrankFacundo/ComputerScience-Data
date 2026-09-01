"""Generates REPORT.md from the model. Run: python3 report.py"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

from leasing_core import (
    BNP_CARS, Common, LeasePolicy, OwnCarPolicy, Car,
    bik_rate, evaluate_company_car, evaluate_own_car, early_exit_cost,
    breakeven_purchase_price, budget_to_car_price, affordable_price,
    annuity_payment, energy_cost_annual,
)
from lux_tax import TaxParams, net_annual, net_of_extra_gross, marginal_rate

E = lambda x: f"{x:,.0f}".replace(",", " ")
E2 = lambda x: f"{x:,.2f}".replace(",", " ")


def build(base_salary=85_000.0, tax_class="1", horizon=36, budget=535.0):
    tax = TaxParams(tax_class=tax_class)
    common = Common(base_gross_salary=base_salary, horizon_months=horizon)
    policy = LeasePolicy(budget_monthly=budget, cash_alternative_gross=budget)
    L = []
    w = L.append

    ref = BNP_CARS[0]
    own_ref = OwnCarPolicy(purchase_price=ref.list_price,
                           consumption_kwh_100km=ref.consumption_kwh_100km)

    a = evaluate_company_car(ref, common, policy, tax)
    b = evaluate_own_car(common, policy, own_ref, tax)
    cash = net_of_extra_gross(base_salary, budget * 12, tax)
    be = breakeven_purchase_price(ref, common, policy, own_ref, tax)

    # ------------------------------------------------------------------
    w(f"# Company car vs cash allowance — BNP Luxembourg")
    w("")
    w(f"*Generated {dt.date.today().isoformat()} by `report.py`. "
      f"All figures are reproducible from `leasing_core.py`.*")
    w("")
    w("---")
    w("")
    w("## The short answer")
    w("")
    w(f"**Take the leasing.** On the reference car (BMW iX2 eDrive20, the cheapest "
      f"on the list) the company car costs you **{E2(a['monthly_out_of_pocket'])} €/month** "
      f"out of pocket. Buying the same car with the cash alternative costs "
      f"**{E2(b['net_cost_nominal']/horizon)} €/month** net of everything, including "
      f"the resale value you keep at the end.")
    w("")
    w(f"Over {horizon} months: **{E(a['net_cost_nominal'])} €** versus "
      f"**{E(b['net_cost_nominal'])} €**. The gap is "
      f"**{E(b['net_cost_nominal'] - a['net_cost_nominal'])} €**.")
    w("")
    w("Three things in your reasoning move the answer, and all three move it "
      "the same way:")
    w("")
    w("1. **You can't deduct 200 €/month of loan payments.** Luxembourg caps "
      f"deductible debt interest at **{E(tax.interest_premium_ceiling)} €/year** "
      "(*intérêts débiteurs*, special expenses) — and that ceiling is *shared* "
      "with your insurance premiums. Capital repayment is never deductible. "
      f"The real saving is about **{E(b['tax_saving_total']/ (horizon/12))} €/year**, "
      f"not 2 400 €/year.")
    w("2. **The company car is barely taxed right now.** An electric company "
      f"car is a benefit in kind of only **0.5 %/month** of list price until the "
      f"end of 2026. On the iX2 that's {E(a['bik_monthly'])} €/month of notional "
      f"income, costing you {E2(a['bik_tax_cost_monthly'])} €/month in tax and "
      "social security. Not zero — but very cheap.")
    w("3. **The 535 € is worth much less as cash than as a car.** "
      f"535 € gross gives you **{E2(cash['net']/12)} € net** "
      f"({cash['keep_ratio']*100:.1f} %), while as a leasing budget it buys "
      f"{E(policy.budget_monthly)} € of car, insurance, maintenance, tyres and "
      "road tax at full value.")
    w("")
    w("Your flexibility argument is the real one, and it survives the maths — "
      "it just isn't worth the gap. See §5.")
    w("")
    w("---")
    w("")

    # ------------------------------------------------------------------
    w("## 1. What 535 € gross is actually worth to you")
    w("")
    w(f"Assumptions: base gross salary **{E(base_salary)} €/year**, tax class "
      f"**{tax_class}**, resident, 2026 rates.")
    w("")
    n = net_annual(base_salary, tax)
    w(f"- Marginal income tax incl. 7 % solidarity surcharge: "
      f"**{marginal_rate(base_salary, tax)*100:.2f} %**")
    w(f"- Employee social security: pension {tax.pension_rate*100:.1f} % + "
      f"sickness {tax.sickness_rate*100:.2f} % + dependency "
      f"{tax.dependency_rate*100:.1f} % = **{(tax.pension_rate+tax.sickness_rate+tax.dependency_rate)*100:.2f} %**")
    w("")
    w("| | € / month | € / year |")
    w("|---|---:|---:|")
    w(f"| Gross allowance | {E2(budget)} | {E(budget*12)} |")
    w(f"| – social security | −{E2(cash['social']/12)} | −{E(cash['social'])} |")
    w(f"| – income tax + solidarity | −{E2(cash['tax']/12)} | −{E(cash['tax'])} |")
    w(f"| **= net in your pocket** | **{E2(cash['net']/12)}** | **{E(cash['net'])}** |")
    w("")
    w(f"You keep **{cash['keep_ratio']*100:.1f} %**. ")
    if tax_class == "1":
        t2 = TaxParams(tax_class="2")
        c2 = net_of_extra_gross(base_salary, budget * 12, t2)
        w(f"Your 335 € estimate corresponds to **tax class 2** (jointly taxed), "
          f"which on this salary gives {E2(c2['net']/12)} € net. In class 1 it is "
          f"{E2(cash['net']/12)} €. Set your real class in the app — it matters.")
    w("")

    # ------------------------------------------------------------------
    w("## 2. What the company car actually costs you")
    w("")
    w("In Luxembourg you are not taxed on the rental the employer pays. You are "
      "taxed on a **benefit in kind (BIK)**: a fixed monthly percentage of the "
      "car's list price when new, options and VAT included, discounts deducted. "
      "It is treated as salary, so it carries both income tax **and** social "
      "security contributions.")
    w("")
    w("| Vehicle type | Registered ≤ 31 Dec 2026 | Registered from 1 Jan 2027 |")
    w("|---|---:|---:|")
    w("| 100 % electric, ≤ 18 kWh/100 km | **0.5 %** | 1.0 % |")
    w("| 100 % electric, > 18 kWh/100 km | **0.6 %** | 1.2 % |")
    w("| Combustion / hybrid | 2.0 % | 2.0 % |")
    w("")
    w("> **This is a dated opportunity.** The 0.5 / 0.6 % rate is locked in for "
      "the whole life of the contract if the car is registered by 31 Dec 2026 — "
      "or by 31 Dec 2027 where the contract was signed before 31 Dec 2026. "
      "Sign in 2026 and you keep the cheap rate for the full 36 months. Sign in "
      "2027 and your BIK **doubles**.")
    w("")
    w(f"### All eight cars, at {E(base_salary)} € salary, class {tax_class}")
    w("")
    w("| Car | Lease €/mo | List price (est.) | BIK rate | BIK €/mo | Tax cost €/mo | Over budget €/mo | **All-in €/mo** |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for car in BNP_CARS:
        r = evaluate_company_car(car, common, policy, tax)
        rows.append((car, r))
        w(f"| {car.name} | {E(car.lease_monthly)} | {E(car.list_price)} | "
          f"{r['bik_rate']*100:.1f} % | {E2(r['bik_monthly'])} | "
          f"{E2(r['bik_tax_cost_monthly'])} | {E2(r['overbudget_monthly'])} | "
          f"**{E2(r['monthly_out_of_pocket'])}** |")
    w("")
    w(f"*All-in includes home charging at {common.energy_price_per_kwh:.2f} €/kWh "
      f"over {E(common.annual_km)} km/year "
      f"(≈ {E2(rows[0][1]['energy_monthly'])} €/mo), which the leasing does not "
      "cover in the default settings. Over-budget top-up is paid from net salary.*")
    w("")
    w("> ⚠️ **List prices are my estimates.** The BIK is a straight percentage of "
      "that number, so it drives everything. Ask BNP's leaser for the *valeur à "
      "neuf TTC* on each quote and put the real figures into the app.")
    w("")
    w("Note the ranking flips: the **GLC at 657 €/mo lease** costs you more than "
      "the iX2 not mainly because of the BIK, but because 122 €/mo of over-budget "
      "comes straight out of your net salary. Staying at or under 535 € is worth "
      "more than shaving the BIK.")
    w("")

    # ------------------------------------------------------------------
    w("## 3. Head to head, same car, same 3 years")
    w("")
    w(f"Reference: **{ref.name}**, list price {E(ref.list_price)} €. "
      f"Option B buys that exact car with a "
      f"{own_ref.loan_term_months}-month loan at {own_ref.loan_rate_apr*100:.1f} % TAEG.")
    w("")
    w("| | A — Company car | B — Cash + buy the same car |")
    w("|---|---:|---:|")
    w(f"| Cash you receive | 0 | +{E(b['cash_net_monthly']*horizon)} |")
    w(f"| Loan payments | 0 | −{E(b['loan_payment_monthly']*min(horizon, own_ref.loan_term_months))} |")
    w(f"| Tax + social security on BIK | −{E(a['bik_tax_cost_monthly']*horizon)} | 0 |")
    w(f"| Insurance, maintenance, tyres, road tax | included | −{E(b['running_monthly']*horizon)} |")
    w(f"| Electricity | −{E(a['energy_monthly']*horizon)} | −{E(b['energy_monthly']*horizon)} |")
    w(f"| Tax saving on deductible interest | 0 | +{E(b['tax_saving_total'])} |")
    w(f"| **Cash out over {horizon} months** | **{E(a['total_nominal_cost'])}** | **{E(b['total_nominal_cost'])}** |")
    w(f"| Car value at month {horizon} ({own_ref.residual_value_pct*100:.0f} %) | 0 | +{E(b['resale_value'])} |")
    w(f"| Loan still outstanding | 0 | −{E(b['loan_balance_at_horizon'])} |")
    w(f"| **= Net cost over {horizon} months** | **{E(a['net_cost_nominal'])}** | **{E(b['net_cost_nominal'])}** |")
    w(f"| Net cost per month | {E2(a['net_cost_nominal']/horizon)} | {E2(b['net_cost_nominal']/horizon)} |")
    w(f"| NPV @ {common.discount_rate_annual*100:.0f} % | {E(a['npv'])} | {E(b['npv'])} |")
    w("")
    w(f"**The company car is {E(b['net_cost_nominal']-a['net_cost_nominal'])} € cheaper "
      f"over {horizon} months for the identical vehicle.**")
    w("")
    w("The reason is not the tax trick — it is that the employer is spending "
      f"**{E(a['employer_spend'])} €** on this car over three years and you are "
      f"taxed on a benefit of only {E(a['bik_monthly']*horizon)} €. If you take "
      f"cash instead, that {E(a['employer_spend'])} € shrinks to "
      f"{E(b['cash_net_monthly']*horizon)} € in your hands, and you then have to "
      "fund the same car — including its depreciation, which is the single "
      "largest line in car ownership and the one the leasing company is "
      "absorbing for you.")
    w("")

    # ------------------------------------------------------------------
    w("## 4. The 200 €/month credit idea, priced properly")
    w("")
    p200 = affordable_price(200.0, own_ref.loan_rate_apr, own_ref.loan_term_months)
    w(f"At {own_ref.loan_rate_apr*100:.1f} % over {own_ref.loan_term_months} months, "
      f"**200 €/month borrows {E(p200)} €**. That is the entire budget — before "
      "insurance, tyres, maintenance and road tax, which the leasing covers and "
      f"which run about {E(own_ref.insurance_annual+own_ref.maintenance_annual+own_ref.tyres_annual+own_ref.road_tax_annual)} €/year on a car of this class.")
    w("")
    w(f"To buy the {ref.name} on the same terms you would pay "
      f"**{E2(annuity_payment(ref.list_price, own_ref.loan_rate_apr, own_ref.loan_term_months))} €/month**, "
      f"not 200 €.")
    w("")
    w("### The deduction that isn't there")
    w("")
    w("| | Your assumption | Luxembourg reality |")
    w("|---|---:|---:|")
    w("| Deductible base | full 200 €/mo payment | interest only |")
    w(f"| Annual amount | 2 400 € | {E(b['interest_paid_horizon']/(horizon/12))} € interest paid… |")
    w(f"| …of which deductible | 2 400 € | **{E(tax.interest_headroom)} €** (ceiling) |")
    w(f"| Cash value per year | ~1 000 € | **{E(b['tax_saving_total']/(horizon/12))} €** |")
    w("")
    w(f"The *intérêts débiteurs* ceiling is {E(tax.interest_premium_ceiling)} €/year "
      "per person (× spouse, × child). It is a **shared** ceiling with your "
      "deductible insurance premiums — if you already claim those, the car loan "
      "adds **nothing**. Set `interest_premium_ceiling_used` in the app to test that.")
    w("")

    # ------------------------------------------------------------------
    w("## 5. Where you are right")
    w("")
    w("### The 3-year lock-in is a real cost")
    w("")
    w("| You leave BNP at… | Penalty (share "
      f"{policy.early_exit_penalty_share*100:.0f} % of remaining rentals) | "
      "Company car total | Still cheaper than buying? |")
    w("|---|---:|---:|:--:|")
    for leave in (6, 12, 18, 24, 30, 36):
        pen = early_exit_cost(ref, policy, leave, horizon)
        tot = a['monthly_out_of_pocket'] * leave + pen
        # buying, pro-rated to the same period
        cb = replace(common, horizon_months=leave)
        bb = evaluate_own_car(cb, policy, own_ref, tax)
        w(f"| month {leave} | {E(pen)} | {E(tot)} | "
          f"{'✅ yes' if tot < bb['net_cost_nominal'] else '❌ no'} "
          f"({E(bb['net_cost_nominal'])} €) |")
    w("")
    w("*Both columns assume the car budget stops when you leave; the buying "
      "column is the same purchase run for a shorter horizon, so its resale "
      "value is higher and its loan balance larger.*")
    w("")
    w("Even leaving at month 6 and eating half the remaining rentals, the leasing "
      "still wins on cost. The lock-in is a **liquidity and optionality** problem, "
      "not a pricing one: it is a lump sum you may have to find at the worst "
      "possible moment — the month you change job.")
    w("")
    w("**Get the exact early-termination clause in writing before signing.** "
      "Ask specifically: (a) what happens if you resign, (b) what happens if you "
      "are made redundant, (c) whether the new employer can take over the "
      "contract, (d) the per-km penalty above the mileage cap. Policies vary "
      "from *employer absorbs everything* to *employee owes 100 % of remaining "
      "rentals*, and the difference is thousands of euros. The default here "
      f"({policy.early_exit_penalty_share*100:.0f} %) is a guess — replace it.")
    w("")
    w("### Ownership really does give you something")
    w("")
    w("- You can sell any day, at any price, to anyone.")
    w("- No mileage cap, no return-condition inspection, no wear-and-tear bill.")
    w("- The car survives a job change.")
    w(f"- At month {horizon} you hold **{E(b['terminal_equity'])} €** of net equity.")
    w("")
    w("None of that is free — the table in §3 prices it at "
      f"{E(b['net_cost_nominal']-a['net_cost_nominal'])} € over three years, "
      f"or about {E(( b['net_cost_nominal']-a['net_cost_nominal'])/horizon)} €/month. "
      "That is what you are paying for flexibility. Whether it is worth it is a "
      "judgement call, not a calculation — but it is a large premium.")
    w("")

    # ------------------------------------------------------------------
    w("## 6. When does buying actually win?")
    w("")
    if be:
        w(f"Break-even: buying beats the company car only if the car you buy costs "
          f"**≤ {E(be)} €**.")
        w("")
        w(f"Above {E(be)} €, the leasing is cheaper. Below it, you are no longer "
          "comparing like with like — you are choosing to drive a much more modest "
          "car and keep the difference. That is a legitimate choice, but it is a "
          "**lifestyle** decision, not an arbitrage.")
    else:
        w("Buying never wins at these parameters, at any purchase price.")
    w("")
    b_af = budget_to_car_price(budget, own_ref, common)
    w(f"### What an all-in {E(budget)} €/month budget buys you as an owner")
    w("")
    w("| Monthly budget | – running costs | → loan payment | Car you can buy "
      f"({own_ref.loan_term_months} mo @ {own_ref.loan_rate_apr*100:.1f} %) |")
    w("|---:|---:|---:|---:|")
    running = (own_ref.insurance_annual + own_ref.maintenance_annual +
               own_ref.tyres_annual + own_ref.road_tax_annual +
               own_ref.inspection_annual) / 12
    for bud in (300, 400, 535, 700, 900, 1100):
        w(f"| {E(bud)} € | −{E2(running)} € | {E2(max(0,bud-running))} € | "
          f"**{E(budget_to_car_price(bud, own_ref, common))} €** |")
    w("")
    w(f"So the honest comparison is: **a {E(ref.list_price)} € BMW iX2 as a company "
      f"car for {E2(a['monthly_out_of_pocket'])} €/month**, versus **a "
      f"{E(b_af)} € car you own for {E(budget)} €/month**. That is roughly a "
      "three-year-old Volkswagen ID.3, a Renault Mégane E-Tech, a MG4, a used "
      "Tesla Model 3 at the high end, or a mid-range petrol hatchback. Nothing "
      "wrong with any of them — but they are not the same product.")
    w("")

    # ------------------------------------------------------------------
    w("## 7. Which car to choose from the BNP list")
    w("")
    ranked = sorted(rows, key=lambda r: r[1]['monthly_out_of_pocket'])
    w("Ranked by what actually leaves your account each month:")
    w("")
    w("| # | Car | All-in €/mo | Why |")
    w("|---:|---|---:|---|")
    for i, (car, r) in enumerate(ranked, 1):
        why = []
        if r['overbudget_monthly'] > 0:
            why.append(f"{E(car.lease_monthly-budget)} € over budget")
        if r['bik_rate'] > 0.005:
            why.append(f"{car.consumption_kwh_100km} kWh/100km → 0.6 % BIK")
        if not why:
            why.append("within budget, 0.5 % BIK")
        w(f"| {i} | {car.name} | **{E2(r['monthly_out_of_pocket'])}** | {', '.join(why)} |")
    w("")
    w("Two structural points:")
    w("")
    w(f"1. **The 18 kWh/100 km line is worth real money.** Cross it and your BIK "
      f"rate goes from 0.5 % to 0.6 % — a 20 % increase, permanent for the "
      "contract. The GLB, GLC and C 400 are on the wrong side of it in my "
      "estimates. Ask for the WLTP combined consumption on the quote; if a car "
      "is at 17.9 vs 18.1, that single number is worth "
      f"{E2(60000*0.001*(1-cash['keep_ratio']))} €/month.")
    w("2. **Going over the 535 € budget is the most expensive thing you can do.** "
      "The top-up comes out of net salary, euro for euro, with no tax shelter at "
      "all. The GLC costs 122 €/month more in lease than the iX2 — and every one "
      "of those euros is post-tax.")
    w("")

    # ------------------------------------------------------------------
    w("## 8. Recommendation")
    w("")
    w("1. **Take the leasing, and sign it in 2026.** The 0.5 % electric BIK rate "
      "is locked for the contract's life if you register by 31 Dec 2026 (or "
      "sign by then and register by 31 Dec 2027). From 2027 it doubles to 1 %. "
      "That deadline is worth more than any car choice on the list.")
    w("2. **Stay at or under the 535 € budget.** The iX2 and iX1 are the only two "
      "that do. Everything above is post-tax money.")
    w("3. **Get three numbers in writing before you sign:** the *valeur à neuf "
      "TTC* (drives your BIK), the WLTP consumption (0.5 % vs 0.6 %), and the "
      "early-termination clause (your entire downside).")
    w("4. **Drop the loan-interest deduction from your reasoning.** It is worth "
      f"about {E(b['tax_saving_total']/(horizon/12))} €/year at best, and zero if "
      "you already use the ceiling on insurance premiums.")
    w("5. **Revisit at renewal, not now.** If you expect to leave BNP within a "
      "year, or you want a cheap car and the cash, buying is defensible — but "
      "then buy a ~15–20 k€ used EV, not a 55 k€ one. Buying the same car as the "
      "leasing is the one option that is clearly dominated.")
    w("")

    # ------------------------------------------------------------------
    w("## Model assumptions")
    w("")
    w("| Parameter | Value | Note |")
    w("|---|---:|---|")
    w(f"| Base gross salary | {E(base_salary)} €/yr | changes the wedge — set yours |")
    w(f"| Tax class | {tax_class} | class 2 keeps ~63 % vs ~50 % in class 1 |")
    w(f"| Horizon | {horizon} months | the leasing term |")
    w(f"| Annual mileage | {E(common.annual_km)} km | |")
    w(f"| Electricity | {common.energy_price_per_kwh:.2f} €/kWh | home charging |")
    w(f"| Loan | {own_ref.loan_rate_apr*100:.1f} % over {own_ref.loan_term_months} mo | LU consumer credit |")
    w(f"| Residual value at {horizon} mo | {own_ref.residual_value_pct*100:.0f} % | premium EVs depreciate fast |")
    w(f"| Insurance / maintenance / tyres | {E(own_ref.insurance_annual)} / {E(own_ref.maintenance_annual)} / {E(own_ref.tyres_annual)} €/yr | owner pays, leasing covers |")
    w(f"| Early-exit penalty | {policy.early_exit_penalty_share*100:.0f} % of remaining rentals | **guess — get the clause** |")
    w(f"| Discount rate | {common.discount_rate_annual*100:.0f} % | |")
    w("")
    w("### Tax and social parameters (Luxembourg, resident employee, 2026)")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Income tax scale | 2025 indexed, 23 bands, 0 → 42 % |")
    w(f"| Solidarity surcharge | 7 % (9 % above {E(tax.solidarity_threshold)} €) |")
    w(f"| Pension (employee) | {tax.pension_rate*100:.1f} % — raised from 8.0 % in 2026 |")
    w(f"| Sickness (employee) | {tax.sickness_rate*100:.2f} % (2.80 % in-kind + 0.25 % cash) |")
    w(f"| Dependency | {tax.dependency_rate*100:.1f} %, no ceiling, not deductible |")
    w(f"| Contribution ceiling | {E(tax.socsec_ceiling_monthly)} €/month |")
    w(f"| Debt-interest deduction | {E(tax.interest_premium_ceiling)} €/yr, shared with insurance premiums |")
    w(f"| Company car BIK | 0.5 / 0.6 % electric to 2026, 1.0 / 1.2 % from 2027, 2 % thermal |")
    w("")
    w("### Sources")
    w("")
    w("- [Reform of company car taxation — Securex](https://securex.lu/en/reform-company-car-taxation/)")
    w("- [Extended benefit-in-kind rate for electric vehicles — Leasys](https://www.leasys.com/lu/english/blog/leasing_extended_benefit_in_kind_rate_for_electric_vehicles)")
    w("- [The tax scale in Luxembourg — taxx.lu](https://taxx.lu/en/tax-return-guide/the-tax-scale-in-luxembourg)")
    w("- [Luxembourg — Individual — Other taxes (social security) — PwC](https://taxsummaries.pwc.com/luxembourg/individual/other-taxes)")
    w("- [Intérêts débiteurs déductibles comme dépenses spéciales — ACD](https://impotsdirects.public.lu/fr/az/i/inter_debit.html)")
    w("- [Tax provisions for company cars for employees (leasing) — Guichet.lu](https://guichet.public.lu/en/citoyens/transport/transports-individuels/voiture-societe-salarie/leasing.html)")
    w("- [Social parameters from 1 January 2026 — Securex](https://securex.lu/en/discover-the-new-social-parameters-applicable-from-january-1st-2026/)")
    w("")
    w("---")
    w("")
    w("*This is a financial model, not tax advice. The list prices are estimates "
      "and the early-exit penalty is a guess; both should be replaced with the "
      "figures on your actual quote before you sign anything. Confirm the "
      "treatment with BNP's HR/payroll or a Luxembourg tax adviser.*")

    return "\n".join(L)


if __name__ == "__main__":
    import sys, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPORT.md")
    md = build()
    with open(out, "w") as f:
        f.write(md + "\n")
    print(f"wrote {out} ({len(md.splitlines())} lines)")
