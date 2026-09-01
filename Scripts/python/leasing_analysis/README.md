# Leasing vs cash allowance — Luxembourg company car

Prices a BNP Luxembourg car policy: take the **leasing** (company car, taxed on
a benefit in kind) or take the **cash alternative** and buy your own car with a
consumer loan.

Read **[REPORT.md](REPORT.md)** for the analysis and the recommendation.

## Files

| File | What it is |
|---|---|
| `lux_tax.py` | Luxembourg payroll engine: 2025/26 tax scale, classes 1 & 2, solidarity surcharge, employee social security with ceilings. Pure stdlib. |
| `leasing_core.py` | Benefit-in-kind rates, loan amortisation, the two scenarios, NPV, break-even solver, the BNP car list. Pure stdlib. |
| `report.py` | Generates `REPORT.md`. |
| `app.py` | Streamlit front-end over the same core. |
| `test_model.py` | 29 sanity checks on the maths. |

## Run

```bash
python3 test_model.py     # verify the model
python3 report.py         # regenerate REPORT.md

pip install -r requirements.txt
streamlit run app.py      # interactive version
```

## Before you trust the euros, replace these

The model is only as good as three numbers it cannot know:

1. **Vehicle list price** (*valeur à neuf TTC*, options included, discounts
   deducted) — the benefit in kind is a straight percentage of it. The prices
   in `BNP_CARS` are **my estimates**. Get them from the leaser's quote.
2. **WLTP consumption** — at or below 18 kWh/100 km the BIK rate is 0.5 %,
   above it 0.6 %. A car sitting near the line is worth checking.
3. **Early-termination clause** — what you owe if you leave BNP mid-contract.
   The 50 % default is a guess and it is your entire downside.

Also set your real **gross salary** and **tax class** in the sidebar: class 2
keeps ~63 % of a gross allowance where class 1 keeps ~50 %, and that ratio is
what the whole comparison turns on.

## Key parameters (Luxembourg, resident employee, 2026)

- Company car BIK: **0.5 % / 0.6 %** per month of list price for electric cars
  registered by 31 Dec 2026 (or contracted by then and registered by
  31 Dec 2027); **1.0 % / 1.2 %** from 2027; **2 %** for combustion/hybrid.
- Employee social security: pension 8.5 % (raised in 2026), sickness 3.05 %,
  dependency 1.4 %; ceiling 13 856.63 €/month, dependency uncapped.
- Deductible debt interest (*intérêts débiteurs*): **672 €/year** per person,
  interest only, **shared** with deductible insurance premiums.

Sources are listed at the bottom of `REPORT.md`.

> Financial model, not tax advice. Confirm with BNP payroll or a Luxembourg
> tax adviser before signing.
