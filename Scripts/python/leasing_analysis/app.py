"""Streamlit front-end. Run:  streamlit run app.py

All maths lives in leasing_core.py / lux_tax.py — this file only binds
widgets to those parameters and draws the results.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from leasing_core import (  # noqa: E402
    BNP_CARS, Car, Common, LeasePolicy, OwnCarPolicy,
    affordable_price, annuity_payment, bik_rate, breakeven_purchase_price,
    budget_to_car_price, early_exit_cost, evaluate_company_car,
    evaluate_own_car,
)
from lux_tax import (  # noqa: E402
    TaxParams, marginal_rate, net_annual, net_of_extra_gross,
)

st.set_page_config(page_title="Leasing vs cash — Luxembourg", layout="wide",
                   page_icon="🚗")

eur = lambda x: f"{x:,.0f} €".replace(",", " ")
eur2 = lambda x: f"{x:,.2f} €".replace(",", " ")

# ==========================================================================
# Sidebar — every assumption is editable
# ==========================================================================
st.sidebar.title("Assumptions")

st.sidebar.header("You")
salary = st.sidebar.number_input(
    "Base gross salary (€/year, excl. car budget)", 20_000, 400_000, 85_000, 1_000,
    help="Drives your marginal rate, which is what the whole comparison turns on.")
tax_class = st.sidebar.selectbox(
    "Tax class", ["1", "2"], index=0,
    help="Class 2 = jointly taxed (married/partnered). Class 1a sits between "
         "the two; use class 1 as the conservative proxy.")

st.sidebar.header("Employer car policy")
budget = st.sidebar.number_input("Monthly leasing budget (€)", 0, 2_000, 535, 5)
cash_alt = st.sidebar.number_input(
    "Cash alternative offered instead (€/month GROSS)", 0, 2_000, 535, 5)
horizon = st.sidebar.slider("Contract length (months)", 12, 60, 36, 6)
signed_2026 = st.sidebar.checkbox(
    "Contract signed / car registered by 31 Dec 2026", True,
    help="Locks the 0.5 % / 0.6 % electric BIK rate for the whole contract. "
         "Uncheck to see the 2027 rates (1.0 % / 1.2 %).")
lease_energy = st.sidebar.checkbox("Leasing includes a charge/fuel card", False)
overbudget_net = st.sidebar.checkbox(
    "Over-budget top-up comes from NET salary", True,
    help="Uncheck if your employer lets you take it from gross.")
underbudget_refund = st.sidebar.checkbox(
    "Unused budget is paid back as cash", False)
penalty_share = st.sidebar.slider(
    "Early-exit penalty (share of remaining rentals you owe)", 0.0, 1.0, 0.5, 0.05,
    help="A guess until you read the contract. Ask HR for the exact clause.")

st.sidebar.header("Usage")
annual_km = st.sidebar.number_input("Annual mileage (km)", 0, 60_000, 15_000, 1_000)
kwh_price = st.sidebar.number_input("Electricity (€/kWh)", 0.0, 1.0, 0.25, 0.01)
discount = st.sidebar.slider("Discount rate (%/yr)", 0.0, 10.0, 3.0, 0.5) / 100

st.sidebar.header("If you buy instead")
loan_rate = st.sidebar.slider("Car loan TAEG (%)", 0.0, 15.0, 5.5, 0.1) / 100
loan_term = st.sidebar.slider("Loan term (months)", 12, 120, 60, 6)
down_payment = st.sidebar.number_input("Down payment (€)", 0, 100_000, 0, 500)
residual = st.sidebar.slider(
    "Resale value at end of horizon (% of purchase price)", 0, 100, 50, 1) / 100
ins = st.sidebar.number_input("Insurance (€/year)", 0, 5_000, 950, 50)
maint = st.sidebar.number_input("Maintenance (€/year)", 0, 5_000, 400, 50)
tyres = st.sidebar.number_input("Tyres (€/year)", 0, 3_000, 300, 50)
roadtax = st.sidebar.number_input("Road tax (€/year)", 0, 2_000, 30, 10)

st.sidebar.header("Tax fine print")
ceiling_used = st.sidebar.number_input(
    "Debt-interest ceiling already used by insurance premiums (€/year)",
    0, 672, 0, 24,
    help="The 672 €/yr ceiling is SHARED between loan interest and deductible "
         "insurance premiums. If you already fill it, your car loan deducts nothing.")

# ==========================================================================
# Assemble
# ==========================================================================
tax = TaxParams(tax_class=tax_class, interest_premium_ceiling_used=ceiling_used)
common = Common(base_gross_salary=salary, horizon_months=horizon,
                annual_km=annual_km, energy_price_per_kwh=kwh_price,
                discount_rate_annual=discount)
policy = LeasePolicy(budget_monthly=budget, cash_alternative_gross=cash_alt,
                     includes_energy=lease_energy,
                     overbudget_from_net=overbudget_net,
                     underbudget_refunded=underbudget_refund,
                     early_exit_penalty_share=penalty_share,
                     contract_signed_by_end_2026=signed_2026)

# ==========================================================================
st.title("🚗 Company car vs cash allowance — Luxembourg")
st.caption("BNP Luxembourg car policy. Every number below recomputes from the "
           "sidebar. Model: `leasing_core.py`, tax engine: `lux_tax.py`.")

# --- car catalogue, editable -----------------------------------------------
st.header("1 · The cars")
st.markdown(
    "**List price drives the benefit in kind.** These are my estimates — edit "
    "the table with the *valeur à neuf TTC* from BNP's quote. Consumption "
    "matters too: the BIK rate steps from 0.5 % to 0.6 % above **18 kWh/100 km**.")

cars_df = pd.DataFrame([{
    "Car": c.name, "Body": c.body, "Lease €/mo": c.lease_monthly,
    "List price € TTC": c.list_price, "kWh/100km": c.consumption_kwh_100km,
    "Electric": c.is_electric,
} for c in BNP_CARS])

edited = st.data_editor(cars_df, num_rows="dynamic", width="stretch",
                        key="cars")

cars = [Car(name=r["Car"], list_price=float(r["List price € TTC"]),
            lease_monthly=float(r["Lease €/mo"]),
            consumption_kwh_100km=float(r["kWh/100km"]),
            is_electric=bool(r["Electric"]), body=str(r.get("Body", "")))
        for _, r in edited.iterrows() if pd.notna(r["Car"])]

if not cars:
    st.stop()

results = [(c, evaluate_company_car(c, common, policy, tax)) for c in cars]

table = pd.DataFrame([{
    "Car": c.name,
    "Lease €/mo": c.lease_monthly,
    "BIK rate": f"{r['bik_rate']*100:.1f} %",
    "BIK €/mo": round(r["bik_monthly"], 2),
    "Tax+social €/mo": round(r["bik_tax_cost_monthly"], 2),
    "Over budget €/mo": round(r["overbudget_monthly"], 2),
    "Energy €/mo": round(r["energy_monthly"], 2),
    "ALL-IN €/mo": round(r["monthly_out_of_pocket"], 2),
    f"Total {horizon} mo": round(r["net_cost_nominal"]),
} for c, r in results]).sort_values("ALL-IN €/mo")

st.dataframe(table, width="stretch", hide_index=True)
st.bar_chart(table.set_index("Car")["ALL-IN €/mo"])

# --- head to head ----------------------------------------------------------
st.header("2 · Head to head")
cheapest = table.iloc[0]["Car"]
names = [c.name for c in cars]
choice = st.selectbox("Reference car", names, index=names.index(cheapest),
                      help="Defaults to the cheapest car for you, all-in.")
car = next(c for c in cars if c.name == choice)
a = evaluate_company_car(car, common, policy, tax)

st.markdown("**What would you buy instead?**")
c1, c2 = st.columns(2)
with c1:
    same = st.checkbox("Buy the same car (like-for-like)", True)
with c2:
    price = st.number_input("Purchase price (€)", 1_000, 300_000,
                            int(car.list_price), 500, disabled=same)
purchase = car.list_price if same else price

own = OwnCarPolicy(purchase_price=purchase, down_payment=down_payment,
                   loan_rate_apr=loan_rate, loan_term_months=loan_term,
                   insurance_annual=ins, maintenance_annual=maint,
                   tyres_annual=tyres, road_tax_annual=roadtax,
                   residual_value_pct=residual, is_electric=car.is_electric,
                   consumption_kwh_100km=car.consumption_kwh_100km)
b = evaluate_own_car(common, policy, own, tax)

m1, m2, m3 = st.columns(3)
m1.metric("A · Company car", eur(a["net_cost_nominal"]),
          f"{eur2(a['monthly_out_of_pocket'])}/mo out of pocket")
m2.metric("B · Cash + buy", eur(b["net_cost_nominal"]),
          f"{eur2(b['net_cost_nominal']/horizon)}/mo net cost")
delta = b["net_cost_nominal"] - a["net_cost_nominal"]
m3.metric("Company car saves you", eur(delta),
          "over the contract" if delta > 0 else "buying wins",
          delta_color="normal" if delta > 0 else "inverse")

cash = net_of_extra_gross(salary, cash_alt * 12, tax)
loan_paid = b["loan_payment_monthly"] * min(horizon, loan_term)
run_tot = b["running_monthly"] * horizon

def _c(x):
    return x if isinstance(x, str) else eur(x)

st.dataframe(pd.DataFrame({
    "": ["Cash received (net)", "Loan payments", "Tax + social on BIK",
         "Insurance / maintenance / tyres / road tax", "Energy",
         "Tax saving on deductible interest",
         f"= Cash out over {horizon} mo", "Car resale value",
         "Loan still outstanding", f"= NET COST over {horizon} mo",
         f"NPV @ {discount*100:.1f} %"],
    "A · Company car": [_c(v) for v in [
        0, 0, -a["bik_tax_cost_monthly"]*horizon, "included",
        -a["energy_monthly"]*horizon, 0, a["total_nominal_cost"], 0, 0,
        a["net_cost_nominal"], a["npv"]]],
    "B · Cash + buy": [_c(v) for v in [
        b["cash_net_monthly"]*horizon, -loan_paid, 0, -run_tot,
        -b["energy_monthly"]*horizon, b["tax_saving_total"],
        b["total_nominal_cost"], b["resale_value"],
        -b["loan_balance_at_horizon"], b["net_cost_nominal"], b["npv"]]],
}), width="stretch", hide_index=True)

st.caption(
    f"Employer spends {eur(a['employer_spend'])} on this leasing over {horizon} "
    f"months. As cash it would reach you as {eur(b['cash_net_monthly']*horizon)} "
    f"({cash['keep_ratio']*100:.1f} % of gross).")

# --- payroll wedge ---------------------------------------------------------
with st.expander("Why 535 € gross is not 535 €"):
    st.markdown(
        f"- Marginal income tax incl. solidarity surcharge: "
        f"**{marginal_rate(salary, tax)*100:.2f} %**\n"
        f"- Social security: pension {tax.pension_rate*100:.1f} % + sickness "
        f"{tax.sickness_rate*100:.2f} % + dependency {tax.dependency_rate*100:.1f} % "
        f"= **{(tax.pension_rate+tax.sickness_rate+tax.dependency_rate)*100:.2f} %**\n"
        f"- Net annual salary at {eur(salary)}: **{eur(net_annual(salary, tax)['net'])}**")
    st.dataframe(pd.DataFrame({
        "": ["Gross allowance", "− social security", "− income tax + solidarity",
             "= net"],
        "€/month": [round(cash_alt, 2), -round(cash["social"]/12, 2),
                    -round(cash["tax"]/12, 2), round(cash["net"]/12, 2)],
        "€/year": [round(cash_alt*12), -round(cash["social"]),
                   -round(cash["tax"]), round(cash["net"])],
    }), width="stretch", hide_index=True)

# --- the deduction ---------------------------------------------------------
st.header("3 · The loan-interest deduction")
d1, d2, d3 = st.columns(3)
d1.metric("Interest paid over horizon", eur(b["interest_paid_horizon"]))
d2.metric("Of which deductible", eur(b["interest_deductible_total"]),
          f"ceiling {eur(tax.interest_headroom)}/yr")
d3.metric("Cash value of the deduction", eur(b["tax_saving_total"]),
          f"{eur(b['tax_saving_total']/(horizon/12))}/yr")
st.info(
    "Luxembourg caps deductible **intérêts débiteurs** at 672 €/year per person "
    "(× spouse, × child), and only the *interest* counts — never capital "
    "repayment. The ceiling is shared with deductible insurance premiums, so if "
    "you already claim those, a car loan deducts nothing. This is the main gap "
    "in the “I'll deduct 200 €/month” reasoning.")

# --- leaving early ---------------------------------------------------------
st.header("4 · What if you leave BNP early?")
rows = []
for leave in range(6, horizon + 1, 6):
    pen = early_exit_cost(car, policy, leave, horizon)
    tot = a["monthly_out_of_pocket"] * leave + pen
    bb = evaluate_own_car(replace(common, horizon_months=leave), policy, own, tax)
    rows.append({"Leave at month": leave, "Penalty €": round(pen),
                 "Company car total €": round(tot),
                 "Buying total €": round(bb["net_cost_nominal"]),
                 "Leasing still cheaper": "yes" if tot < bb["net_cost_nominal"] else "no"})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.warning(
    "The penalty share above is a **guess**. Get the early-termination clause in "
    "writing: what happens if you resign, if you are made redundant, whether a "
    "new employer can take the contract over, and the per-km charge above the "
    "mileage cap. This is your entire downside and the model cannot know it.")

# --- break-even ------------------------------------------------------------
st.header("5 · What can you buy instead?")
be = breakeven_purchase_price(car, common, policy, own, tax)
if be:
    st.success(
        f"**Break-even: {eur(be)}.** Buying only beats the company car if the car "
        f"you buy costs less than that. Above it, the leasing wins. Below it, you "
        f"are choosing a more modest car and keeping the difference — a lifestyle "
        f"call, not an arbitrage.")
else:
    st.success("At these parameters the company car wins at any purchase price.")

st.markdown(f"An owner pays running costs out of the same budget "
            f"({eur2((ins+maint+tyres+roadtax+own.inspection_annual)/12)}/mo), "
            f"so the loan only gets what is left:")
st.dataframe(pd.DataFrame([{
    "All-in budget €/mo": bud,
    "− running costs": round((ins+maint+tyres+roadtax+own.inspection_annual)/12, 2),
    "→ loan payment €/mo": round(max(0, bud-(ins+maint+tyres+roadtax+own.inspection_annual)/12), 2),
    f"Car you can buy ({loan_term} mo @ {loan_rate*100:.1f} %)":
        round(budget_to_car_price(bud, own, common)),
} for bud in [200, 300, 400, budget, 700, 900, 1100]]),
    width="stretch", hide_index=True)

st.caption(
    f"For reference, buying the {car.name} outright on this loan costs "
    f"{eur2(annuity_payment(car.list_price-down_payment, loan_rate, loan_term))}/month.")

# --- BIK deadline ----------------------------------------------------------
st.header("6 · The 2026 deadline")
alt = replace(policy, contract_signed_by_end_2026=not signed_2026)
a_alt = evaluate_company_car(car, common, policy=alt, tax=tax)
k1, k2 = st.columns(2)
k1.metric(f"BIK rate {'≤ 2026' if signed_2026 else 'from 2027'}",
          f"{a['bik_rate']*100:.1f} %", f"{eur2(a['monthly_out_of_pocket'])}/mo all-in")
k2.metric(f"BIK rate {'from 2027' if signed_2026 else '≤ 2026'}",
          f"{a_alt['bik_rate']*100:.1f} %",
          f"{eur2(a_alt['monthly_out_of_pocket'])}/mo all-in")
st.info(
    "Electric company cars keep the 0.5 % / 0.6 % benefit-in-kind rate if the "
    "car is registered by **31 Dec 2026**, or the contract is signed by then and "
    "the car registered by 31 Dec 2027. From 2027 the rates double to 1.0 % / "
    "1.2 %. The rate is fixed for the life of the contract, so signing in 2026 "
    f"is worth {eur(abs(a_alt['net_cost_nominal']-a['net_cost_nominal']))} over "
    f"{horizon} months on this car.")

st.divider()
st.caption(
    "Financial model, not tax advice. List prices are estimates and the "
    "early-exit penalty is a guess — replace both with the figures on your "
    "actual quote. Confirm treatment with BNP payroll or a Luxembourg tax adviser.")
