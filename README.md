# PhoebeCaster

Code and data for the demo paper *PhoebeCaster: Configuration-to-Performance
Forecasting for Fast-Moving AI Agents on Multi-Model, Hybrid Workloads*.

PhoebeCaster sizes the database of an AI-agent application: from a serial
query probe and two anchor runs, it predicts completion time for any core count,
hyper-threading setting and concurrency level, and returns the core count
closest to a target or "none fits".

## Layout

| Path | Contents |
|---|---|
| `src/` | the model, the core scan and the evaluation code |
| `paper/` | one script per group of numbers or figures in the paper |
| `data/` | measurements and model inputs (see below) |
| `results/` | outputs of the `paper/` scripts |
| `webui/` | the demo interface |
| `harness/` | the measurement harness (documents the protocol; needs the instrumented PostgreSQL) |

## Reproduce the paper's numbers

Needs Python 3.9+, `numpy`, `scipy`, `matplotlib`. Run from any directory:

```
python3 paper/demo_failmap.py        # 224 scored targets         -> results/demo_failmap.json
python3 paper/paper_numbers_v2.py    # Table 2, dataset, abstract -> results/paper_numbers_v2.json
python3 paper/rev1_numbers.py        # Sections 2-4               -> results/rev1_numbers.json
python3 paper/make_fig_dilation.py   # Figure 2                   -> ./fig_dilation.pdf
```

Run `demo_failmap.py` before `rev1_numbers.py`, which reads its output.

## Run the demo

```
pip install django gunicorn
cd webui && ./start_server.sh      # http://127.0.0.1:8900
```

Recommendations, panels and the failure map work from the bundled
measurements. The "measure live" button needs the measurement setup below.

The command-line tool gives the same answers:

```
python3 src/advise.py 125 64 --workload test2      # 125 s at 64 agents
```

## Data

| Path | Contents |
|---|---|
| `data/rc3/`, `data/rc4/` | training runs on 10 BIRD databases: one `point.json` per run (db_time split into four components, completion time, per-query times) |
| `data/testv2/` | the two held-out workloads (`test` = bal_test, `test2`), 196 configurations |
| `data/testv2_live/` | configurations measured live outside the grid |
| `data/offline_*.json`, `data/*_label_noise.json` | serial-probe results (per-query offline cost) |
| `data/anchors_*_v2.json` | the two anchor runs of each held-out workload |
| `data/train_set_v2.json`, `data/r_model.json`, `data/ht_gain_fit.json` | fitted inputs of the prior model |
| `data/questions/`, `data/*_trace.json` | BIRD questions per database and the replayed LLM query rounds |
| `data/*sql_screen.json` | LLM-written query rounds excluded from replay (see below) |

Notes:

- Query rounds that run longer than 1 s on their own are excluded from replay
  (`data/*sql_screen.json`): three in the training workloads, all with wrong
  results, and one in the held-out workloads (bal_test question 900086), with a
  correct result.
- bal_test was re-measured on 2026-10-29. The earlier runs replayed question
  900086 because the replay read only the training screen; `harness/workload.py`
  now reads every screen file and drops a question whose only usable round was
  screened out. Six test2 configurations re-measured alongside agree with the
  earlier runs within 4.0% in completion time.
- `european_football_2` is excluded from training (`src/badpoints.py`,
  `EXCLUDED_LIBS`): one of its screened-out query rounds was replayed anyway
  through a fallback path in `harness/workload.py`, so all of its runs contain a
  ~21 s query that should not be there (the fallback is fixed since).

## Measurement setup (not included)

Measurements ran on PostgreSQL 17 with the transactional HNSW index
`cognitivehnsw`, which is not part of this release; the scripts in `harness/`
document the protocol and cannot be run without it. Core budgets are applied
with `taskset` (`SUDO_PW` must be exported) and memory with cgroups. The BIRD
databases are not redistributed here; download the BIRD dev set
(CC BY-SA 4.0) and set `BIRD_DIR`.

## License

Code: MIT (see `LICENSE`). The questions in `data/questions/` derive from BIRD
(CC BY-SA 4.0).
