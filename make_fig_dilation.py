#!/usr/bin/env python3
"""图 3: 四个分量的每步耗时随并发膨胀多少(相对 8 并发), 两个留出负载, 16 核 / HT 开。
数据: bird-agent/rev1_numbers.py → rev1_numbers.json["dilation"](testv2 实测)。
形式: 要表达"各分量放大倍数不同", 折线 + 双对数轴; 两个负载各一小图, 共用 y 轴。
配色同 make_fig_crossover.py(dataviz 参考调色板 slot 1-4, 已过验证器; 青/黄对比度 WARN → 线端直标)。
用法: python3 make_fig_dilation.py → fig_dilation.pdf"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rev1_numbers.json")))["dilation"]
COMP = [("sql", "analytical SQL", "#2a78d6"),
        ("tp_txn", "write-back", "#1baf7a"),
        ("vec_value", "business-data search", "#eda100"),
        ("vec_mem", "memory search", "#eb6834")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7.2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.62), sharey=True)
for ax, (wl, name) in zip(axs, (("bal_test", "bal_test"), ("test2", "test2"))):
    d = R[wl]; x = d["agents"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, linewidth=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ends = []
    for k, lab, col in COMP:
        ax.plot(x, d[k], color=col, linewidth=1.4, marker="o", markersize=2.6,
                markeredgecolor="white", markeredgewidth=0.4, label=lab)
        ends.append([d[k][-1], d[k][-1]])
    # 线端直标(墨色), 在对数空间里把太近的标签推开
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][1] / ends[i - 1][1] < 1.28: ends[i][1] = ends[i - 1][1] * 1.28
    for v, y in ends:
        ax.text(250, y, f"{v:.1f}×", va="center", ha="left", fontsize=6.4, color=INK)
    ax.set_xticks([8, 16, 32, 64, 224]); ax.set_xticklabels(["8", "16", "32", "64", "224"])
    ax.minorticks_off()
    ax.set_xlim(7, 224)
    ax.set_title(name, fontsize=7.2, color=INK, pad=2, family="monospace")
    ax.set_xlabel("concurrent agents", color=INK2, labelpad=1)
    ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)
axs[0].set_yticks([1, 2, 4, 8, 16]); axs[0].set_yticklabels(["1×", "2×", "4×", "8×", "16×"])
axs[0].set_ylim(0.85, 22)
axs[0].set_ylabel("time per step\n(relative to 8 agents)", color=INK2, labelpad=1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.6, handlelength=1.6,
           columnspacing=1.2, borderaxespad=0.1, bbox_to_anchor=(0.54, 1.02))
fig.subplots_adjust(left=0.17, right=0.92, bottom=0.2, top=0.74, wspace=0.42)
fig.savefig("fig_dilation.pdf"); fig.savefig("fig_dilation_preview.png", dpi=220)
