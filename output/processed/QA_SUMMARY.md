# Data QA Summary

## consumption

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.13)
- **extreme_outliers**: OK

## fai_ytd_growth

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.18)
- **extreme_outliers**: OK

## industrial_value_added

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.47)
- **extreme_outliers**: OK

## m2_m1_m0

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.11)
- **extreme_outliers**: OK

## new_home_price_index

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.12)
- **extreme_outliers**: OK

## new_loans

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.20)
- **extreme_outliers**: OK

## pmi

- **duplicates(series,date)**: FAIL: 12 duplicates
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.16)
- **extreme_outliers**: WARN: 1 points with |z|>8

## ppi

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.33)
- **extreme_outliers**: OK

## tsf_increment

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.12)
- **extreme_outliers**: OK

## usdcny

- **duplicates(series,date)**: OK
- **missing_values**: OK
- **date_parse**: OK
- **frequency_infer**: OK: OK: freq not clearly monthly (monthly-ish ratio=0.20)
- **extreme_outliers**: OK
