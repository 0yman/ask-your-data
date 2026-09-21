# Agent evaluation

- Model: `gemini-3.1-flash-lite`
- Questions: **22** (20 answerable, 2 unanswerable by design)
- Generated: 2026-09-21 14:47 UTC

## Headline

| Metric | Value |
|---|---|
| Answer figure coverage (end to end) | **0.944** |
| Execution accuracy (strict set equality) | 0.750 |
| Gold coverage (relaxed) | 0.900 |
| Correctly declined (unanswerable) | 1.000 |

## By difficulty

| Difficulty | Execution accuracy |
|---|---|
| easy | 1.000 |
| hard | 0.625 |
| medium | 0.714 |

## Cost and behaviour

| Metric | Value |
|---|---|
| Mean steps per question | 3.96 |
| Mean queries per question | 1.00 |
| Self-corrections triggered | 0 |
| Mean latency | 28.8s |
| Prompt tokens | 121,086 |
| Output tokens | 4,638 |

## Per question

| id | difficulty | exec | figures | steps | retries | stop reason |
|---|---|---|---|---|---|---|
| q01 | easy | yes | 1.00 | 3 | 0 | final_answer |
| q02 | easy | yes | 1.00 | 5 | 0 | final_answer |
| q03 | easy | yes | 1.00 | 5 | 0 | final_answer |
| q04 | easy | yes | 1.00 | 2 | 0 | final_answer |
| q05 | easy | yes | 1.00 | 2 | 0 | final_answer |
| q06 | medium | yes | 1.00 | 5 | 0 | final_answer |
| q07 | medium | yes | 1.00 | 2 | 0 | final_answer |
| q08 | medium | yes | 1.00 | 2 | 0 | final_answer |
| q09 | medium | yes | 1.00 | 5 | 0 | final_answer |
| q10 | medium | **no** | 0.00 | 4 | 0 | final_answer |
| q11 | medium | yes | 1.00 | 5 | 0 | final_answer |
| q12 | medium | covers | 1.00 | 5 | 0 | final_answer |
| q13 | hard | yes | 1.00 | 6 | 0 | final_answer |
| q14 | hard | covers | 1.00 | 5 | 0 | final_answer |
| q15 | hard | covers | 1.00 | 5 | 0 | final_answer |
| q16 | hard | **no** | 0.88 | 2 | 0 | final_answer |
| q17 | hard | yes | 1.00 | 2 | 0 | final_answer |
| q18 | hard | yes | 1.00 | 2 | 0 | final_answer |
| q19 | hard | yes | 1.00 | 2 | 0 | final_answer |
| q20 | hard | yes | 1.00 | 6 | 0 | final_answer |
| q21 | trap | declined ok | - | 4 | 0 | final_answer |
| q22 | trap | declined ok | - | 8 | 0 | final_answer |

## Failures

**q10** — What percentage of container movements involve hazardous cargo?

```sql
SELECT 100.0 * SUM(CASE WHEN ct.is_hazardous THEN fcm.container_count ELSE 0 END) / SUM(fcm.container_count) AS hazardous_percentage FROM fact_container_movement AS fcm JOIN dim_cargo_type AS ct ON fcm.cargo_type_key = ct.cargo_type_key LIMIT 1000
```

> Approximately 4.78% of container movements involve hazardous cargo.

**q12** — Which vessel type waits longest before berthing on average?

```sql
SELECT v.vessel_type, AVG(vc.waiting_hours) AS avg_waiting_hours FROM fact_vessel_call AS vc JOIN dim_vessel AS v ON vc.vessel_key = v.vessel_key GROUP BY v.vessel_type ORDER BY avg_waiting_hours DESC LIMIT 1000
```

> The vessel type that waits the longest before berthing on average is the Post-Panamax, with an average waiting time of 6.73 hours. The other vessel types have average waiting times of 6.55 hours for Neo-Panamax, 6.54 hours for Panamax, and 6.47 hours for Feeder vessels.

**q14** — Which berth underperforms most relative to how many cranes it has, measured as moves per hour per crane?

```sql
SELECT b.berth_code, b.crane_count, AVG(v.moves_per_hour) AS avg_moves_per_hour, AVG(v.moves_per_hour) / b.crane_count AS moves_per_hour_per_crane FROM fact_vessel_call AS v JOIN dim_berth AS b ON v.berth_key = b.berth_key GROUP BY b.berth_code, b.crane_count ORDER BY moves_per_hour_per_crane ASC LIMIT 1000
```

> The berth that underperforms most relative to its crane count is B07, which has 3 cranes and achieves an average of 6.06 moves per hour per crane.

**q15** — For each terminal, what share of vessel calls waited more than 12 hours before berthing?

```sql
SELECT b.terminal, COUNT(*) AS total_calls, SUM(CASE WHEN v.waiting_hours > 12 THEN 1 ELSE 0 END) AS long_wait_calls, (SUM(CASE WHEN v.waiting_hours > 12 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS share_long_wait_percent FROM fact_vessel_call AS v JOIN dim_berth AS b ON v.berth_key = b.berth_key GROUP BY b.terminal LIMIT 1000
```

> The share of vessel calls that waited more than 12 hours before berthing, broken down by terminal, is as follows:
- Alexandria Container Terminal: 15.91% (377 out of 2,370 calls)
- East Harbour Quay: 15.30% (43 out of 281 calls)
- El Dekheila Terminal: 15.94% (346 out of 2,170 calls)

**q16** — Compare total TEU per year and show the year-over-year change.

```sql
SELECT d.year, SUM(f.teu) AS total_teu FROM fact_container_movement AS f JOIN dim_date AS d ON f.date_key = d.date_key GROUP BY d.year ORDER BY d.year LIMIT 1000
```

> The total TEU handled per year and the year-over-year changes are as follows:
- In 2023, the total TEU was 1,947,148.
- In 2024, the total TEU was 1,907,902, representing a decrease of 39,246 TEU (approximately -2.02%) compared to 2023.
- In 2025, the total TEU was 1,987,349, representing an increase of 79,447 TEU (approximately +4.16%) compared to 2024.

