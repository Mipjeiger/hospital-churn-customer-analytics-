Engineering report — Hospital Care Analytics Dashboard

Files Changed
  Created/overwritten: /Users/miftahhadiyannoor/Documents/Hospital_healthcare/develop/streamlit/app.py (0 bytes -> 78K, single-file production dashboard)
  No other files modified. Database at develop/database/fix_final_hospital_database.parquet is read-only.

Dataset Summary (fix_final_hospital_database.parquet, inspected before coding)
  Rows: 10,000
  Columns: 20
  Column names: patient_id, name, age, arrival_date, departure_date, service, satisfaction, month, week, available_beds, patients_request, patients_admitted, patients_refused, patient_satisfaction, staff_morale, total_staff, staff_present, staff_absens (typo -> aliased as staff_absent), length_of_stay, patient_churn
  Types: ints/floats + datetime64[arrival/departure] + object ids/names/services
  Missing values: 0
  Duplicate patient_id: 0 | Duplicate rows: 0
  Date range: arrival 2025-01-01 to 2025-12-31, departure 2025-01-03 to 2026-01-13
  Services (4): emergency 2694, surgery 2557, general_medicine 2402, ICU 2347
  Churn: 6028 churned (60.3%), 3972 retained
  Age: 0-100, mean 45.4 | LOS: 1-14 d, mean 7.5 | satisfaction 35-100 mean 79.0 | patient_satisfaction 34.9-100 mean 78.0 | staff_morale 5-100 mean 73.8 | available_beds 1-80 mean 32.8
  Notes: LOS matches arrival/departure diff exactly; no negative LOS; departure < arrival = 0

Analytics Implemented (all live from filtered df, safe-division, no NaN leaks)
  Reusable metrics: calculate_churn_rate, calculate_admission_rate, calculate_refusal_rate, calculate_staff_attendance_rate, calculate_capacity_pressure, calculate_risk_score
  Derived: arrival_year/month/week/day/dow/is_weekend, age_group (6 bands), satisfaction_bin (5 bands), capacity_pressure = requests/(beds+1) -> Low<1/Moderate1-2/High2-4/Critical>4, risk_score 0-100 = 0.40dissat +0.25LOSnorm +0.20refusal +0.15morale -> Low<=30/Medium31-60/High>60, staff_attendance_rate, admission/refusal row rates, Service Health Score = 100(0.35sat/100 +0.25(1-churn)+0.20admission+0.20*attendance)
  KPIs: total/churned/rate, satisfaction, LOS, morale, admission/refusal, beds, attendance, capacity pressure (Overview); age/LOS/sat/churn variants per page

UI Implemented
  Theme: Inter font, navy #0F2A44 + teal #0E7C7B palette, KPI cards, section headers, insight boxes, sidebar navy
  Sidebar navigation (10): Overview, Patient Analytics, Patient Management, Churn Analytics, Hospital Operations, Staff Analytics, Service Performance, Operational Risk, Management Insights, Data Explorer
  Global filters (sidebar): arrival_date range, service multi-select (dynamic), age slider, churn All/Churned/Retained — all pages use filtered df; empty filter shows warning, not crash
  Visuals (Plotly): Overview — volume line, churn trend, service bar, churn-by-service bar, satisfaction vs churn, capacity area, requests vs admissions grouped bar; Patient Analytics — age histogram, age-group bar+line, LOS histogram + box by churn, sat histogram, service donut; Patient Management — searchable table + attention list + CSV download; Churn — donut, bar, age/sat bars, box, scatter (sampled 2000), Service x Age heatmap + n-flag table; Hospital Ops — requests/admissions grouped bar, admission/refusal by service, beds line, demand vs capacity line, pressure distribution; Staff — presence trend, absence hist, morale trend, morale vs sat scatter+ols, absence vs churn bar, correlation heatmap; Service Performance — full performance table + health score ranking + 5 ranking cards; Operational Risk — risk distribution, risk by service, risk vs sat scatter, capacity by service, top-risk patients; Management Insights — 8 auto-generated insights + KPI summary; Data Explorer — quality KPIs, validation warnings, tabs for preview/types/missing/descriptive + correlation heatmap, filtered download
  Privacy: patient names only on Patient Management / Risk drill-down; executive views aggregated; disclaimers on every relevant page
  Caching: @st.cache_data for load_data; derived features computed once

Dependencies
  No new packages. Uses existing: streamlit 1.51.0, pandas 2.1.4, plotly 5.24.1, pyarrow 24.0.0, numpy, statsmodels 0.14.6 (for trendline, already installed)

Validation
  Dataset: parquet loads via robust path resolver (checks file, cwd, rglob), schema check, datetime coercion, zero missing confirmed
  Filters: date/service/age/churn tested empty and single-service; empty returns warning table-free paths guarded
  Metrics: safe_div prevents ZeroDivision/NaN; admission/refusal with zero requests -> 0%; churn on empty -> 0; LOS negative -> 0; correlations use only existing columns
  Edge cases tested: empty df, zero requests, n<30 groups flagged with caution, sampled scatter for perf, py_compile + streamlit health 200 on :8509
  Run: streamlit headless started, health /_stcore/health = ok, no startup errors

Limitations / Data Issues (surfaced in Data Explorer, not silently fixed)
  2166 records have patients_refused > patients_request (~21.7%)
  8766 records have staff_present+staff_absent != total_staff; staff_present exceeds total_staff in many rows (max 79.6 vs total max 39) — suggests synthetic/aggregated fields, not literal roster
  satisfaction vs patient_satisfaction uncorrelated (r -0.004) despite similar names — treated as distinct signals
  churn 60.3% is unusually high; likely synthetic
  No diagnosis columns present — no clinical inference attempted

Run Command
  cd /Users/miftahhadiyannoor/Documents/Hospital_healthcare/develop
  streamlit run streamlit/app.py
  (or from repo root: streamlit run develop/streamlit/app.py)
  App loaded successfully — verified via health endpoint 200 on localhost:8509.

File ready at: /Users/miftahhadiyannoor/Documents/Hospital_healthcare/develop/streamlit/app.py