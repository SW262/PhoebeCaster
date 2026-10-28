# PhoebeCaster

Code and data for the demo paper *PhoebeCaster: Configuration-to-Performance
Forecasting for Fast-Moving AI Agents on Multi-Model, Hybrid Workloads*.

PhoebeCaster sizes the database of an AI-agent application: from a serial
query probe and two anchor runs, it predicts completion time for any core count,
hyper-threading setting and concurrency level, and returns the core count
closest to a target or "none fits".

## Reproduce the paper's numbers

Needs Python 3.9+, `numpy`, `scipy`, `matplotlib`. Every number in the paper is
recomputed from the measurements in this repository:

```
python3 demo_failmap.py        # 224 scored targets         -> demo_failmap.json
python3 paper_numbers_v2.py    # Table 2, dataset, abstract -> paper_numbers_v2.json
python3 rev1_numbers.py        # Sections 2-4 additions     -> rev1_numbers.json
python3 make_fig_dilation.py   # Figure 2                   -> fig_dilation.pdf
```

## Run the demo

```
pip install django gunicorn
cd webui && ./start_server.sh      # http://127.0.0.1:8900
```

Recommendations, panels and the failure map work from the bundled
measurements. The "measure live" button needs the measurement setup below.

## Layout

| Path | Contents |
|---|---|
| `rc3/`, `rc4/` | training runs on 10 BIRD databases: one `point.json` per run (db_time split into four components, completion time, per-query times) |
| `testv2/` | the two held-out workloads (`test` = bal_test, `test2`), 196 configurations |
| `testv2_live/` | configurations measured live outside the grid |
| `offline_*.json`, `*_label_noise.json` | serial-probe results (per-query offline cost) |
| `phoebe_predictor.py`, `sql_model2.py`, `ns_model2.py` | prior model and calibration |
| `phoebe_cli.py`, `phoebe_advisor.py`, `advise.py` | core scan and recommendation |
| `eval_*.py`, `score_rc.py`, `badpoints.py` | data loading and scoring |
| `workload.py`, `collect_bird.sh`, `live_measure.sh`, `llm.py` | measurement harness |
| `webui/` | the demo interface |

## Data notes

- `sql_screen.json`: LLM-written query rounds that run longer than 1 s on their own
  are excluded from replay; every excluded round also returns a wrong result.
- `european_football_2` is excluded from training (`badpoints.EXCLUDED_LIBS`): one of
  its screened-out query rounds was replayed anyway through a fallback path in
  `workload.py`, so all of its runs contain a ~21 s query that should not be there.

## Measurement setup (not included)

Measurements ran on PostgreSQL 17 with the transactional HNSW index
`cognitivehnsw`, which is not part of this release; the harness scripts are
included to document the protocol and cannot be run without it. Core budgets
are applied with `taskset` (`SUDO_PW` must be exported) and memory with cgroups.
The BIRD databases are not redistributed here; download the BIRD dev set
(CC BY-SA 4.0) and set `BIRD_DIR`.

## License

Code: MIT (see `LICENSE`). The questions in `pdb_*_questions.json` derive from BIRD (CC BY-SA 4.0).
