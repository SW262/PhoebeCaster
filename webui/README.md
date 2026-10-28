# PhoebeCaster web UI

Django + gunicorn, one template. Binds to 127.0.0.1 by default.

```
./start_server.sh            # start on 127.0.0.1:8900
./start_server.sh 8901 4     # other port / worker count
./start_server.sh --stop     # stop (reads webui.pid)
```

| Endpoint | What it returns |
|---|---|
| `GET /` | dashboard |
| `GET /api/advise?target=&agents=&workload=&max_cores=` | recommendation (`advise.py --json`) |
| `GET /api/panels?workload=&agents=` | core sweep and db_time breakdown (`demo_panels.py`) |
| `GET /api/failmap` | the 224 scored targets (`demo_failmap.json`) |
| `POST /api/measure` | live run of one configuration (`live_measure.sh`; needs the instrumented PostgreSQL) |
