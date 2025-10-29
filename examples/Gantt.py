# gantt_compare_weeks.py
# Project Gantt: Plan vs Actual with weekly axis & labels
# pip install plotly kaleido  (kaleido 可选，仅用于导出PNG)

import pandas as pd
import plotly.express as px

# === Week -> Date helper ===
PROJECT_START = pd.Timestamp("2025-08-04")  # Week 1 Monday (保留Week 1坐标，但无任务)
def week_to_date(week: int) -> pd.Timestamp:
    # week=1 -> start date; inclusive
    return PROJECT_START + pd.Timedelta(days=(week - 1) * 7)

def week_span(s: int, e: int):
    # 返回 [start_date, end_date_exclusive]，x_end 用“独占右端”更好看
    start = week_to_date(s)
    end_exclusive = week_to_date(e) + pd.Timedelta(days=7)
    return start, end_exclusive

# ============= 关键改动：Kickoff & Setup 从 Week 1 → Week 2 =============
# === Plan (严格按报告周次，并将Kickoff移到Week 2) ===
plan_rows = [
    ("Kickoff & Setup", *week_span(2, 2), "Phase A: Setup", "plan", ""),  # 改到Week 2
    ("Literature & Methodology", *week_span(2, 3), "Phase B: Research", "plan", ""),
    ("Prototype v1 (Wrapper API)", *week_span(3, 5), "Phase C: Dev", "plan", ""),
    ("API Wrapper Improvement (v1.1)", *week_span(3, 5), "Phase C: Dev", "plan", ""),
    ("Visualization Module v1 – Basic", *week_span(3, 5), "Phase C: Dev", "plan", ""),
    ("Visualization Module v1 – Extended", *week_span(3, 5), "Phase C: Dev", "plan", ""),
    ("Model Development – Iteration v2.1", *week_span(3, 6), "Phase C: Dev", "plan", ""),
    ("Model Development – Iteration v2.2", *week_span(3, 6), "Phase C: Dev", "plan", ""),
    ("Visualization & Testing (v2.3, two phases)", *week_span(7, 11), "Phase D: Test/Vis", "plan", ""),
    ("Unit/Integration/Regression Testing", *week_span(7, 11), "Phase D: Test/Vis", "plan", ""),
    ("Progress Report", *week_span(8, 9), "Phase D: Test/Vis", "plan", ""),
    ("Pre-release Deployment", *week_span(8, 10), "Phase E: Deploy/Docs", "plan", ""),
    ("Draft Documentation", *week_span(8, 10), "Phase E: Deploy/Docs", "plan", ""),
    ("Finalisation", *week_span(12, 13), "Phase F: Finalise", "plan", ""),
    ("Final Report", *week_span(13, 13), "Phase F: Finalise", "plan", ""),
    ("Buffer (Planned)", *week_span(8, 8), "Buffers", "plan", ""),
    ("Buffer (Planned)", *week_span(12, 12), "Buffers", "plan", ""),
]

# === Actual (与本次汇报一致；Kickoff 同步移至 Week 2) ===
actual_rows = [
    # 按时
    ("Kickoff & Setup (Actual)", *week_span(2, 2), "Phase A: Setup", "on-time", "On time"),  # 改到Week 2
    ("Literature & Methodology (Actual)", *week_span(2, 3), "Phase B: Research", "on-time", "On time"),
    ("Progress Report (Actual)", *week_span(8, 9), "Phase D: Test/Vis", "on-time", "On time"),
    ("Finalisation (Actual)", *week_span(12, 13), "Phase F: Finalise", "on-time", "On time"),
    ("Final Report (Actual)", *week_span(13, 13), "Phase F: Finalise", "on-time", "On time"),

    # 延期/偏移（Dev & Vis & Deploy 有所后移/重叠）
    ("Prototype v1 (Wrapper API) (Actual)", *week_span(3, 5), "Phase C: Dev", "on-time", "On time"),
    ("API Wrapper Improvement (v1.1) (Actual)", *week_span(4, 6), "Phase C: Dev", "delayed", "Shifted"),
    ("Visualization Module v1 – Basic (Actual)", *week_span(3, 5), "Phase C: Dev", "on-time", "On time"),
    ("Visualization Module v1 – Extended (Actual)", *week_span(4, 6), "Phase C: Dev", "delayed", "Shifted"),
    ("Model Dev – Iteration v2.1 (Actual)", *week_span(4, 7), "Phase C: Dev", "delayed", "Shifted"),
    ("Model Dev – Iteration v2.2 (Actual)", *week_span(5, 8), "Phase C: Dev", "delayed", "Shifted/Extended"),
    ("Visualization & Testing (v2.3, Actual)", *week_span(9, 13), "Phase D: Test/Vis", "delayed", "Shifted"),
    ("Unit/Integration/Regression (Actual)", *week_span(9, 13), "Phase D: Test/Vis", "delayed", "Shifted"),
    ("Pre-release Deployment (Actual)", *week_span(9, 11), "Phase E: Deploy/Docs", "delayed", "Shifted"),
    ("Draft Documentation (Actual)", *week_span(9, 11), "Phase E: Deploy/Docs", "delayed", "Shifted"),

    # 缓冲被占用
    ("Buffer used – Week 8", *week_span(8, 8), "Buffers", "buffer", "Buffer used"),
    ("Buffer used – Week 12", *week_span(12, 12), "Buffers", "buffer", "Buffer used"),
]

df_plan = pd.DataFrame(plan_rows, columns=["Task","Start","Finish","Group","Status","Label"])
df_actual = pd.DataFrame(actual_rows, columns=["Task","Start","Finish","Group","Status","Label"])
df_plan["Type"] = "Plan"
df_actual["Type"] = "Actual"

# 合并
df = pd.concat([df_plan, df_actual], ignore_index=True)

# Group 顺序
phase_order = [
    "Phase A: Setup","Phase B: Research","Phase C: Dev",
    "Phase D: Test/Vis","Phase E: Deploy/Docs","Phase F: Finalise","Buffers"
]
df["Group"] = pd.Categorical(df["Group"], categories=phase_order, ordered=True)

# 颜色（无橙色）
color_map = {
    "plan":   "#1f77b4",  # 蓝（计划）
    "on-time":"#2ca02c",  # 绿（按时）
    "delayed":"#d62728",  # 红（延期/后移）
    "buffer": "#7f7f7f"   # 灰（缓冲/占用）
}

# 画图
fig = px.timeline(
    df, x_start="Start", x_end="Finish", y="Task",
    color="Status", color_discrete_map=color_map,
    hover_data={"Type":True,"Group":True,"Start":True,"Finish":True,"Status":True}
)

# 条内标签（计划不显示，实际显示）
fig.update_traces(text=df["Label"], textposition="inside", insidetextanchor="middle")

# 横轴：按周显示 Week 1..13（Week 1 将无任务条，但作为参照保留）
tick_vals = [week_to_date(w) for w in range(1, 14)]
tick_text = [f"Week {w}" for w in range(1, 14)]
fig.update_xaxes(
    tickvals=tick_vals,
    ticktext=tick_text,
    title_text="Weeks",
    showgrid=True
)

fig.update_yaxes(title_text="Tasks (Plan vs Actual)")

fig.update_layout(
    title="Project Timeline — Plan vs Actual (Weeks, differences highlighted)",
    bargap=0.25, height=950,
    legend_title_text="Status",
    template="plotly_white"
)

fig.show()

# 导出
fig.write_html("gantt_compare_plan_vs_actual_weeks.html")
try:
    fig.write_image("gantt_compare_plan_vs_actual_weeks.png", scale=2)
except Exception as e:
    print("PNG 导出失败（如需导出请安装 kaleido）：", e)
print("Saved: gantt_compare_plan_vs_actual_weeks.html")
