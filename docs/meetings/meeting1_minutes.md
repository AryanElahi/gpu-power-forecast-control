# Meeting 1 minutes — end of Sprint 1 (proposal and data decisions)

Packet: https://claude.ai/code/artifact/69a5bf15-6eb4-4cc0-8652-4d85f434fde3

## 1. Date and attendees

- Date: [add date, early October 2026]
- Attendees: [supervisor], Aryan Faghirollahy

## 2. Presented

- Proposal (four research questions, 8-week plan)
- Data pipeline result: 2.2 TiB of MIT Supercloud telemetry reduced to a 1-minute GPU power series
- Key EDA figures and the data-quality report
- Ten data decisions for approval

## 3. Supervisor feedback

- The meeting went well; the supervisor left the data-science choices to the student. [add any specific comments]

## 4. Decisions

| # | Decision | Outcome |
|---|---|---|
| 1 | Analysis window 4 March – 30 September 2021 | Approved |
| 2 | Relative ramp definition: abs(ΔP) / L24 above the training p95 | Approved; no ramp label when L24 has fewer than 720 valid minutes |
| 3 | Maintenance event window (23:00 the day before to the end of the day after) | Approved |
| 4 | Same rule for the unscheduled outage of 29 March | Approved |
| 5 | New York wall clock; daylight-saving hour excluded | Approved |
| 6 | Five-minute dip kept in the target; minute-mod-5 feature | Approved |
| 7 | Covariates | Changed: main comparison univariate; covariates a separate arm, used after their gain is confirmed on validation (confirmed in Week 3) |
| 8 | Hour-of-day features yes, weekday no | Approved |
| 9 | Scaling | Changed: Min-Max on train vs window-local scaling, chosen on validation (local scaling chosen in Week 3) |
| 10 | PM100 as a second dataset | Approved with condition: schema check first; deferred |
| — | 10-second supplementary experiment | Dropped (would need the raw data again) |
| — | sMAPE | Kept |

## 5. Next steps

- Sprint 2: classical and foundation benchmark, frozen test protocol (done, see Meeting 2).
