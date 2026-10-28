# Feature Glossary (generated from the study's field definitions)

Field names are the names used by the original data source. A field marked (derived) was computed by the study from the source fields listed with it.

## Taiwan credit-card default data (UCI #350)

Outcome: Default payment (Yes = 1, No = 0) as documented by UCI #350; the widely used copy names it 'default payment next month'. Horizon: the payment due after the September 2005 statement (next month); not 12 months. Amounts are in NT dollars. Data cover April-September 2005.

Repayment-status codes: -1 = paid duly; 1 = payment delay of one month; 2 = two months; ... 9 = nine months or more. **-2 and 0 occur in the data but are not defined by UCI**. Secondary sources interpret them, but the data provider does not, so an explanation must not state a meaning for them. A value of 0 is a recorded value, not a missing one. There is no PAY_1 in the source data.

| Field | Meaning | Unit / coding |
|---|---|---|
| LIMIT_BAL | Amount of the given credit, including the individual consumer's credit and family (supplementary) credit | NT dollars |
| EDUCATION | Education level of the card holder | code |
| MARRIAGE | Marital status of the card holder | code |
| AGE | Age of the card holder | years |
| PAY_0 | Repayment status in September 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT1 | Amount of bill statement in September 2005 | NT dollars |
| PAY_AMT1 | Amount paid in September 2005 (UCI: amount of previous payment) | NT dollars |
| PAY_2 | Repayment status in August 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT2 | Amount of bill statement in August 2005 | NT dollars |
| PAY_AMT2 | Amount paid in August 2005 (UCI: amount of previous payment) | NT dollars |
| PAY_3 | Repayment status in July 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT3 | Amount of bill statement in July 2005 | NT dollars |
| PAY_AMT3 | Amount paid in July 2005 (UCI: amount of previous payment) | NT dollars |
| PAY_4 | Repayment status in June 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT4 | Amount of bill statement in June 2005 | NT dollars |
| PAY_AMT4 | Amount paid in June 2005 (UCI: amount of previous payment) | NT dollars |
| PAY_5 | Repayment status in May 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT5 | Amount of bill statement in May 2005 | NT dollars |
| PAY_AMT5 | Amount paid in May 2005 (UCI: amount of previous payment) | NT dollars |
| PAY_6 | Repayment status in April 2005 | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| BILL_AMT6 | Amount of bill statement in April 2005 | NT dollars |
| PAY_AMT6 | Amount paid in April 2005 (UCI: amount of previous payment) | NT dollars |
| pay_max (derived) | Largest numeric repayment-status code across the 6 monthly statuses PAY_0, PAY_2 ... PAY_6 (April-September 2005). A value of 1 or more is the longest recorded payment delay; how the undocumented codes -2 and 0 rank against -1 (paid duly) is not documented | repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented) |
| pay_sum_late (derived) | Number of months, out of the 6 monthly statuses PAY_0, PAY_2 ... PAY_6, with a payment delay (code 1 or more) | months (0-6) |
| util_ratio (derived) | BILL_AMT1 divided by LIMIT_BAL: the September 2005 bill as a share of the credit limit | ratio (1.0 = bill equals the limit; negative for a credit balance; can exceed 1) |
| pay_ratio (derived) | PAY_AMT1 divided by BILL_AMT1 (amount paid in September 2005 divided by the September 2005 bill). UCI calls PAY_AMT the 'amount of previous payment', so this is not documented as the share of that bill that was paid | ratio; missing when the bill is 0; can be negative (credit balance) or above 1 |

## Home Credit default risk data

Outcome: Payment difficulties (1): late payment of more than X days on at least one of the first Y instalments of the loan; X and Y are not disclosed by the data provider. Horizon: the first Y instalments of the loan (Y not disclosed); not 12 months. The currency of amounts is not documented. DAYS_ fields count days relative to the application (negative = before it); for example DAYS_BIRTH = -12,000 means about 32.9 years old.

Only fields that can appear in the evidence are listed in full below; any other field is described in the evidence line itself.

| Field | Meaning | Unit / coding |
|---|---|---|
| NAME_CONTRACT_TYPE | Identification if loan is cash or revolving |  |
| FLAG_OWN_CAR | Flag if the client owns a car | Y/N |
| FLAG_OWN_REALTY | Flag if client owns a house or flat | Y/N |
| CNT_CHILDREN | Number of children the client has | count |
| AMT_INCOME_TOTAL | Income of the client | currency amount (currency not documented by the data provider) |
| AMT_CREDIT | Credit amount of the loan | currency amount (currency not documented by the data provider) |
| AMT_ANNUITY | Loan annuity | currency amount (currency not documented by the data provider) |
| AMT_GOODS_PRICE | For consumer loans it is the price of the goods for which the loan is given | currency amount (currency not documented by the data provider) |
| NAME_TYPE_SUITE | Who was accompanying client when he was applying for the loan |  |
| NAME_INCOME_TYPE | Clients income type (businessman, working, maternity leave,) |  |
| NAME_EDUCATION_TYPE | Level of highest education the client achieved |  |
| NAME_FAMILY_STATUS | Family status of the client |  |
| NAME_HOUSING_TYPE | What is the housing situation of the client (renting, living with parents, ...) |  |
| REGION_POPULATION_RELATIVE | Normalized population of region where client lives (higher number means the client lives in more populated region) | normalised score (scale not documented) |
| DAYS_BIRTH | Client's age in days at the time of application Note: time only relative to the application. | days relative to the application (negative = before the application) |
| DAYS_EMPLOYED | How many days before the application the person started current employment Note: The value 365243 appears in the data as a placeholder; it is not documented by the data provider and is not a real employment duration. | days relative to the application (negative = before the application) |
| DAYS_REGISTRATION | How many days before the application did client change his registration Note: time only relative to the application. | days relative to the application (negative = before the application) |
| DAYS_ID_PUBLISH | How many days before the application did client change the identity document with which he applied for the loan Note: time only relative to the application. | days relative to the application (negative = before the application) |
| OWN_CAR_AGE | Age of client's car |  |
| FLAG_MOBIL | Did client provide mobile phone (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| FLAG_EMP_PHONE | Did client provide work phone (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| FLAG_WORK_PHONE | Did client provide home phone (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| FLAG_CONT_MOBILE | Was mobile phone reachable (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| FLAG_PHONE | Did client provide home phone (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| FLAG_EMAIL | Did client provide email (1=YES, 0=NO) | flag (1 = yes, 0 = no) |
| OCCUPATION_TYPE | What kind of occupation does the client have |  |
| CNT_FAM_MEMBERS | How many family members does client have | count |
| REGION_RATING_CLIENT | Our rating of the region where client lives (1,2,3) |  |
| REGION_RATING_CLIENT_W_CITY | Our rating of the region where client lives with taking city into account (1,2,3) |  |
| WEEKDAY_APPR_PROCESS_START | On which day of the week did the client apply for the loan |  |
| HOUR_APPR_PROCESS_START | Approximately at what hour did the client apply for the loan Note: rounded. |  |
| REG_REGION_NOT_LIVE_REGION | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) | flag (1 = yes, 0 = no) |
| REG_REGION_NOT_WORK_REGION | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) | flag (1 = yes, 0 = no) |
| LIVE_REGION_NOT_WORK_REGION | Flag if client's contact address does not match work address (1=different, 0=same, at region level) | flag (1 = yes, 0 = no) |
| REG_CITY_NOT_LIVE_CITY | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) | flag (1 = yes, 0 = no) |
| REG_CITY_NOT_WORK_CITY | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) | flag (1 = yes, 0 = no) |
| LIVE_CITY_NOT_WORK_CITY | Flag if client's contact address does not match work address (1=different, 0=same, at city level) | flag (1 = yes, 0 = no) |
| ORGANIZATION_TYPE | Type of organization where client works |  |
| EXT_SOURCE_1 | Normalized score from external data source | normalised score between 0 and 1 (direction not documented) |
| EXT_SOURCE_2 | Normalized score from external data source | normalised score between 0 and 1 (direction not documented) |
| EXT_SOURCE_3 | Normalized score from external data source | normalised score between 0 and 1 (direction not documented) |
| OBS_30_CNT_SOCIAL_CIRCLE | How many observation of client's social surroundings with observable 30 DPD (days past due) default | count |
| DEF_30_CNT_SOCIAL_CIRCLE | How many observation of client's social surroundings defaulted on 30 DPD (days past due) | count |
| OBS_60_CNT_SOCIAL_CIRCLE | How many observation of client's social surroundings with observable 60 DPD (days past due) default | count |
| DEF_60_CNT_SOCIAL_CIRCLE | How many observation of client's social surroundings defaulted on 60 (days past due) DPD | count |
| DAYS_LAST_PHONE_CHANGE | How many days before application did client change phone | days relative to the application (negative = before the application) |
| FLAG_DOCUMENT_2 | Did client provide document 2 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_3 | Did client provide document 3 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_4 | Did client provide document 4 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_5 | Did client provide document 5 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_6 | Did client provide document 6 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_7 | Did client provide document 7 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_8 | Did client provide document 8 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_9 | Did client provide document 9 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_10 | Did client provide document 10 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_11 | Did client provide document 11 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_12 | Did client provide document 12 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_13 | Did client provide document 13 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_14 | Did client provide document 14 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_15 | Did client provide document 15 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_16 | Did client provide document 16 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_17 | Did client provide document 17 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_18 | Did client provide document 18 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_19 | Did client provide document 19 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_20 | Did client provide document 20 | flag (1 = yes, 0 = no) |
| FLAG_DOCUMENT_21 | Did client provide document 21 | flag (1 = yes, 0 = no) |
| AMT_REQ_CREDIT_BUREAU_HOUR | Number of enquiries to Credit Bureau about the client one hour before application | number of enquiries |
| AMT_REQ_CREDIT_BUREAU_DAY | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) | number of enquiries |
| AMT_REQ_CREDIT_BUREAU_WEEK | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) | number of enquiries |
| AMT_REQ_CREDIT_BUREAU_MON | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) | number of enquiries |
| AMT_REQ_CREDIT_BUREAU_QRT | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) | number of enquiries |
| AMT_REQ_CREDIT_BUREAU_YEAR | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) | number of enquiries |
| ratio_credit_income (derived) | AMT_CREDIT divided by AMT_INCOME_TOTAL | ratio |
| ratio_annuity_income (derived) | AMT_ANNUITY divided by AMT_INCOME_TOTAL | ratio |
| credit_term (derived) | AMT_CREDIT divided by AMT_ANNUITY (number of annuity payments needed to repay the credit amount; not the contractual term) | ratio (count of annuity payments) |
| days_employed_pct (derived) | DAYS_EMPLOYED divided by DAYS_BIRTH (share of life spent in current employment) Note: Meaningless when DAYS_EMPLOYED holds the 365243 placeholder. | ratio |
| bureau_n (derived) | Number of the client's credits from other institutions reported to the Credit Bureau | count |
| bureau_days_credit_mean (derived) | Mean, over bureau credits, of days before the current application that the client applied for that credit | days relative to the application (negative = before) |
| bureau_overdue_max (derived) | Largest number of days past due on any bureau credit at the time of the current application | days past due |
| bureau_sum (derived) | Sum of AMT_CREDIT_SUM (credit amount) over the client's bureau credits; credits with no recorded amount count as 0; amounts in different currencies are added as recorded | currency amount (currency not documented by the data provider) |
| bureau_debt (derived) | Sum of AMT_CREDIT_SUM_DEBT (recorded current debt) over the client's bureau credits; credits with no recorded debt count as 0, negative values (credit balances) are included, and amounts in different currencies are added as recorded | currency amount (currency not documented by the data provider) |
| bureau_active_n (derived) | Number of the client's bureau credits with status Active | count |

Building/housing fields ending _AVG, _MODE or _MEDI are normalised statistics about the building where the client lives (documented only as 'normalized').

## Reading the evidence

- A categorical field is shown with its recorded value and the documented label, e.g. `EDUCATION = 2 (university)`. The field name alone never tells you the value.
- `not recorded` means the source has no value for this borrower. It is not zero.
- **Risk score / raw log-odds margin**: the model's internal score before it is converted to a probability. Attribution numbers live on this scale and are not percentages.
- **Calibrated probability of default (PD)**: the only percentage an explanation may state as the default estimate.
