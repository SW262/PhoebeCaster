"""PhoebeCaster 演示服务的最小 Django 配置。

有意不装 DB / admin / auth / sessions: 这个服务没有自己的状态, 所有数据都是
实时从 bird-agent 的产物里读的。没有 DB 就没有 migrate, 也就不会因为 sqlite
锁之类的事情影响到正在跑的采集。
"""
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY = "phoebe-demo-not-a-secret"      # 本机只读演示服务, 不处理任何凭据
DEBUG = False
ALLOWED_HOSTS = ["*"]
INSTALLED_APPS = ["django.contrib.staticfiles"]
MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]
ROOT_URLCONF = "phoebeweb.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [os.path.join(BASE_DIR, "templates")],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": []},
}]
WSGI_APPLICATION = "phoebeweb.wsgi.application"
DATABASES = {}
STATIC_URL = "/static/"
USE_TZ = False
# 必须显式设成本机时区。Django 不设 TIME_ZONE 时默认 America/Chicago, 而且它会
#   直接改进程的 os.environ["TZ"] —— 结果 time.strftime / datetime.now 整体偏 16 小时,
#   页面上的 "最后更新" 和 ETA 全是错的, 还错得很像真的。
TIME_ZONE = os.environ.get("PHOEBE_TZ") or (
    open("/etc/timezone").read().strip() if os.path.exists("/etc/timezone")
    else os.path.realpath("/etc/localtime").split("zoneinfo/")[-1])
LANGUAGE_CODE = "zh-hans"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
