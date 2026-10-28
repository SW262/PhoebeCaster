#!/usr/bin/env python3
# 按 question_id 重测离线逐条代价(串行、无并发、6 遍取后 5 遍中位 —— 与 measure-offline 同口径)
# 目的: 为"与文献同口径的单查询 Q-error"提供可配对的离线代价 c_i。
import json,sys,time,os
import psycopg2
D=os.path.dirname(os.path.abspath(__file__))
wl, trace, dsn = sys.argv[1], sys.argv[2], sys.argv[3]
T=json.load(open(os.path.join(D,trace)))
conn=psycopg2.connect(dsn); conn.autocommit=True; cur=conn.cursor()
cur.execute("SET statement_timeout=60000")
out={}; t0=time.time(); nq=0; nerr=0
for q in T:
    qid=q.get("qid"); db=q.get("db")
    for rd in (q.get("rounds") or []):
        sql=rd.get("sql")
        if not sql: continue
        key=f"{qid}:{rd.get('round',1)}"
        ts=[]
        try:
            cur.execute(f'SET search_path TO "{db}", public')
            for i in range(6):
                s=time.perf_counter(); cur.execute(sql); cur.fetchall()
                ts.append((time.perf_counter()-s)*1000)
        except Exception as e:
            conn.rollback(); nerr+=1; continue
        ts=sorted(ts[1:])                      # 丢第一次(冷), 后 5 遍取中位
        out[key]=round(ts[len(ts)//2],3); nq+=1
print(f"{wl}: {nq} 条 SQL 测完, {nerr} 条失败, 耗时 {time.time()-t0:.1f}s")
# 每题汇总(多轮取各轮之和/中位均可, 这里保留逐轮, 另给每题中位)
byq={}
for k,v in out.items():
    qid=k.split(":")[0]; byq.setdefault(qid,[]).append(v)
json.dump({"cost_by_round":out,
           "cost_by_qid":{k:round(sorted(v)[len(v)//2],3) for k,v in byq.items()}},
          open(os.path.join(D,f"offline_byqid_{wl}.json"),"w"),indent=1)
print(f"  -> offline_byqid_{wl}.json  ({len(byq)} 个 qid)")
