import json, os, subprocess, sys
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.urls import path
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

BIRD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import status as st

WORKLOADS = ("bal_test", "test2")
# 训练网格实际采过的并发档。不在表里的值不接受 —— 给出去就是外推, 而外推从未验证。
AGENTS = (8, 16, 32, 64, 96, 128, 160, 192, 224)


def dashboard(request):
    # 首屏状态服务端直出: 打开就有数, 不先闪一屏 "读取中…"。
    # 取不到也不能让页面打不开 —— 前端拿到 null 会退回轮询。
    try:
        init = json.dumps(st.snapshot(), ensure_ascii=False)
    except Exception:
        init = "null"
    return render(request, "dashboard.html", {"initial": init})


@require_GET
def api_headline(request):
    """验证口径的两个数。读 validation_headline.json —— 由 make_headline.py 实跑生成,
    里面带重算命令。不在页面里写死数字: advise.py 的 docstring 曾经写 87.9%,
    今天实跑是 89.3%, 数据加过点之后旧数就不对了。"""
    p = os.path.join(BIRD, "validation_headline.json")
    if not os.path.exists(p):
        return JsonResponse({}, status=404)
    return JsonResponse(json.load(open(p)))


@require_GET
def api_status(request):
    return JsonResponse(st.snapshot())


@require_GET
def api_advise(request):
    """推荐配置。直接调 advise.py --json —— CLI 和网页共用同一条路径,
    数对不上这种事从根上不会发生。永远不传 --measure: 采集在跑, 网页不许开新负载。"""
    try:
        target = float(request.GET.get("target", ""))
        agents = int(request.GET.get("agents", ""))
    except ValueError:
        return HttpResponseBadRequest("target must be seconds and agents an integer")
    wl = request.GET.get("workload", "test2")
    if wl not in WORKLOADS:
        return HttpResponseBadRequest(f"workload must be one of {WORKLOADS}")
    if agents not in AGENTS:
        return HttpResponseBadRequest(f"agents must be a level measured in the training grid: {AGENTS}")
    if not (1 <= target <= 1e6):
        return HttpResponseBadRequest("target out of range")
    try:
        mc = int(request.GET.get("max_cores", 16))
    except ValueError:
        mc = 16
    mc = max(1, min(mc, 16))
    cmd = [sys.executable, "advise.py", str(target), str(agents),
           "--workload", wl, "--max-cores", str(mc), "--json"]
    r = subprocess.run(cmd, cwd=BIRD, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        return JsonResponse({"error": (r.stderr or r.stdout)[-800:]}, status=500)
    return JsonResponse(json.loads(r.stdout))


@require_GET
def api_panels(request):
    """三个解释面板(测量流水线 / 零实测对比 / 分量拆解)。demo_panels.py 与 advise.py
    同走 phoebe_cli._load, 逐档预测值一致(2026-10-21 核对过)。"""
    wl = request.GET.get("workload", "test2")
    try:
        agents = int(request.GET.get("agents", ""))
        mc = max(1, min(int(request.GET.get("max_cores", 16)), 16))
    except ValueError:
        return HttpResponseBadRequest("agents must be an integer")
    if wl not in WORKLOADS or agents not in AGENTS:
        return HttpResponseBadRequest("bad workload or agents")
    r = subprocess.run([sys.executable, "demo_panels.py", "--workload", wl, "--agents", str(agents),
                        "--max-cores", str(mc)], cwd=BIRD, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        return JsonResponse({"error": (r.stderr or r.stdout)[-800:]}, status=500)
    return JsonResponse(json.loads(r.stdout))


@require_GET
def api_failmap(request):
    """224 个目标逐个对错。读 demo_failmap.json(python3 demo_failmap.py 生成, 合计 assert = 189/224)。"""
    p = os.path.join(BIRD, "demo_failmap.json")
    if not os.path.exists(p):
        return JsonResponse({}, status=404)
    return JsonResponse(json.load(open(p)))


LIVE_STATUS = os.path.join(BIRD, "testv2_live", "status.json")


@csrf_exempt                    # 只绑 127.0.0.1, 页面自己发的 POST
@require_POST
def api_measure(request):
    """现场实测一个点: 起 live_measure.sh(单点, testv2 同定义, 落 testv2_live/, 不进论文数字)。
    一次只跑一个; 有采集在跑、磁盘 < 400G、PG 未就绪都由脚本自己拒绝并写进 status.json。"""
    try:
        wl = request.POST.get("workload", "")
        srv, th, agents = (int(request.POST.get(k, "")) for k in ("srv", "th", "agents"))
    except ValueError:
        return HttpResponseBadRequest("srv, th, agents must be integers")
    if wl not in WORKLOADS or agents not in AGENTS or th not in (1, 2) or not (1 <= srv <= 16):
        return HttpResponseBadRequest("bad parameters")
    pf = os.path.join(BIRD, "live_measure.pid")
    try:
        if os.path.exists(pf) and os.path.exists(f"/proc/{int(open(pf).read.split[0])}"):
            return JsonResponse({"error": "another live measurement is running"}, status=409)
    except (ValueError, IndexError):
        pass
    os.makedirs(os.path.dirname(LIVE_STATUS), exist_ok=True)
    json.dump({"state": "starting", "msg": "starting", "workload": wl, "srv": srv, "th": th, "agents": agents},
              open(LIVE_STATUS, "w"))
    subprocess.Popen(["./live_measure.sh", wl, str(srv), str(th), str(agents)], cwd=BIRD,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return JsonResponse({"ok": True})


@require_GET
def api_measure_status(request):
    try:
        return JsonResponse(json.load(open(LIVE_STATUS)))
    except Exception:
        return JsonResponse({"state": "idle"})


urlpatterns = [
    path("api/measure", api_measure),
    path("api/measure_status", api_measure_status),
    path("api/panels", api_panels),
    path("api/failmap", api_failmap),
    path("api/advise", api_advise),
    path("api/headline", api_headline),
    path("api/status", api_status),
    path("", dashboard, name="dashboard"),
]
