"""
MatGen-Eq 智能材料发现系统
增强版：明亮主题、完整参数面板、实时日志、可视化增强、卡片审查界面
"""
import sys
import json
import zipfile
import io
import time
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from config import (
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH, BASE_MODEL_PATH,
)
import backend.db as db

# ── 页面设置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MatGen-Eq | 智能材料发现系统",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧪",
)

# ── 明亮主题 CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8edf2 100%);
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0d2137 100%);
        border-right: 2px solid #2d5a8a;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #7ec8e3 !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #b8d4e8 !important;
    }

    /* 标题样式 */
    h1 {
        color: #1e3a5f !important;
        font-weight: 900 !important;
        font-size: 2.4rem !important;
        border-bottom: 3px solid #4a90d9;
        padding-bottom: 0.4rem;
    }
    h2 {
        color: #2d5a8a !important;
        font-weight: 700 !important;
    }
    h3 {
        color: #3a6ea5 !important;
        font-weight: 600 !important;
    }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f0f7;
        border-radius: 8px 8px 0 0;
        color: #2d5a8a;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a90d9 !important;
        color: white !important;
        border-bottom: 3px solid #1e3a5f;
    }

    /* 按钮样式 - 主要按钮 */
    div.stButton > button[kind="primary"],
    div.stButton > button {
        background: linear-gradient(90deg, #4a90d9 0%, #2d5a8a 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        width: 100%;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 6px rgba(74, 144, 217, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #5ba0e9 0%, #3d6a9a 100%);
        box-shadow: 0 6px 12px rgba(74, 144, 217, 0.4);
        transform: translateY(-1px);
    }

    /* Metric 卡片 */
    [data-testid="stMetricValue"] {
        color: #1e3a5f !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5a7fa5 !important;
        font-weight: 500 !important;
    }

    /* 成功/警告/错误提示 */
    .success-box, .warning-box, .error-box {
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .success-box {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .warning-box {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .error-box {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        color: #721c24;
    }

    /* 卡片容器 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e8f0;
    }
    .card-header {
        background: linear-gradient(90deg, #4a90d9 0%, #2d5a8a 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        margin: -1.2rem -1.2rem 0.8rem -1.2rem;
    }

    /* 表格样式 */
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a8a 100%) !important;
        color: white !important;
        font-weight: 700;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #f0f4f8 !important;
    }
    .stDataFrame tr:hover {
        background-color: #dce8f5 !important;
    }

    /* 滚动日志区域 */
    .log-container {
        background: #1a1a2e;
        color: #00ff88;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
        line-height: 1.6;
    }
    .log-timestamp { color: #888; }
    .log-info { color: #00ccff; }
    .log-warn { color: #ffaa00; }
    .log-error { color: #ff4444; }

    /* 审查卡片 */
    .review-card {
        background: white;
        border-radius: 12px;
        border: 2px solid #e0e8f0;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    .review-card:hover {
        border-color: #4a90d9;
        box-shadow: 0 4px 12px rgba(74, 144, 217, 0.2);
    }
    .review-card-filtered {
        border-left: 5px solid #28a745;
    }
    .review-card-rejected {
        border-left: 5px solid #dc3545;
    }
    .review-card-pending {
        border-left: 5px solid #ffc107;
    }

    /* 状态标签 */
    .state-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .state-generated { background: #d1ecf1; color: #0c5460; }
    .state-predicted { background: #d4edda; color: #155724; }
    .state-filtered_in { background: #c3e6cb; color: #155724; }
    .state-filtered_out { background: #f8d7da; color: #721c24; }
    .state-validated { background: #d1ecf1; color: #0c5460; }
    .state-rejected { background: #f8d7da; color: #721c24; }

    /* 分隔线 */
    hr { border: none; border-top: 1px solid #c8dce8; margin: 1rem 0; }

    /* 进度条 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4a90d9 0%, #2d5a8a 100%);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #e8f0f7;
        border-radius: 8px;
        font-weight: 600;
        color: #1e3a5f;
    }

    /* 下载按钮特殊样式 */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #28a745 0%, #1e7e34 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #34bd5a 0%, #2e9e44 100%);
    }

    /* 隐藏默认的Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

db.init_db()

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def make_uuid_short(uuid_str: str) -> str:
    """返回UUID的短格式（8字符）"""
    return uuid_str[:8].upper() if uuid_str else "N/A"

def state_badge(state: str) -> str:
    """生成状态徽章HTML"""
    badge_class = f"state-{state}"
    labels = {
        "generated": "已生成",
        "rejected_precheck": "校验失败",
        "predicted": "已预测",
        "filtered_in": "入围",
        "filtered_out": "淘汰",
        "validated": "已确认",
        "rejected": "已驳回",
        "exported_for_training": "已回流",
    }
    return f'<span class="state-badge {badge_class}">{labels.get(state, state)}</span>'

def show_batch_table(samples: list, show_uuid: bool = False, target_dg_h: float = -0.2) -> None:
    """展示样本表格（含偏差列）"""
    if not samples:
        st.info("暂无数据。")
        return
    rows = []
    for s in samples:
        dg_h = s.get("dg_h")
        deviation = round(abs(dg_h - target_dg_h), 4) if dg_h is not None else None
        row = {
            "ID": s["id"],
            "UUID": make_uuid_short(s.get("uuid", "")),
            "时间": (s["created_at"] or "")[:19],
            "提示词摘要": (s["prompt"] or "")[:40] + ("…" if len(s.get("prompt", "") or "") > 40 else ""),
            "ΔG_H (eV)": f"{dg_h:.4f}" if dg_h is not None else "—",
            f"|ΔG_H - {target_dg_h}|": f"{deviation:.4f}" if deviation is not None else "—",
            "通过筛选": "✓" if s.get("in_filter") else "✗",
            "复核": s.get("human_status", "pending"),
            "验证": s.get("validation_status", "pending"),
            "状态": s.get("current_state", "generated"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.map(lambda v: "color: #28a745; font-weight: bold" if v == "✓" else ("color: #dc3545" if v == "✗" else ""), subset=["通过筛选"]),
        use_container_width=True,
        hide_index=True,
    )

def build_log_line(level: str, msg: str) -> str:
    """生成带时间戳的日志行"""
    ts = datetime.now().strftime("%H:%M:%S")
    cls = f"log-{level}"
    return f'<span class="log-timestamp">[{ts}]</span> <span class="{cls}">{msg}</span>'

def log_message(level: str, msg: str) -> str:
    return build_log_line(level, msg)

# ── 侧边栏参数配置 ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 系统参数配置")
    st.divider()

    st.markdown("### 🎯 目标 ΔG_H 设置")
    target_dg_h = st.number_input(
        "目标 ΔG_H (eV)",
        value=-0.2,
        step=0.05,
        format="%.2f",
        help="HER 理想值约为 -0.2 eV",
    )
    tolerance = st.number_input(
        "容差 ± (eV)",
        value=0.30,
        min_value=0.01,
        max_value=2.0,
        step=0.05,
        format="%.2f",
        help="筛选区间：目标 ± 容差",
    )

    st.divider()
    st.markdown("### 🧪 元素组成限定")

    # 可选元素（对应论文中从Ir、Pd、Pt、Rh、Ru选择）
    available_elements = ["Ir", "Pd", "Pt", "Rh", "Ru", "Cu", "Fe", "Co", "Ni"]
    default_elements = st.multiselect(
        "允许的元素种类",
        options=available_elements,
        default=["Ir", "Pd", "Pt", "Rh", "Ru"],
        help="将在生成提示词中体现元素限定",
    )

    st.divider()
    st.markdown("### ⚡ 生成参数")

    max_new_tokens_side = st.number_input(
        "max_new_tokens", min_value=256, max_value=4096,
        value=DEFAULT_MAX_NEW_TOKENS, step=128,
    )
    temperature_side = st.number_input(
        "temperature", min_value=0.0, max_value=2.0,
        value=DEFAULT_TEMPERATURE, step=0.05, format="%.2f",
    )
    top_p_side = st.number_input(
        "top_p", min_value=0.0, max_value=1.0,
        value=DEFAULT_TOP_P, step=0.05, format="%.2f",
    )

    st.divider()
    st.markdown("### 🔧 批处理设置")
    batch_size = st.number_input(
        "子批次大小", min_value=1, max_value=50, value=10, step=1,
        help="每个子批次内的候选结构串行生成",
    )
    num_candidates_side = st.number_input(
        "每条提示词生成候选数", min_value=1, max_value=10, value=1, step=1,
    )

    st.divider()
    st.markdown("### 🖥️ 设备设置")

    # 检测可用设备
    import torch
    device_options = ["auto"]
    device_labels = {"auto": "自动检测"}
    if torch.cuda.is_available():
        device_options.append("cuda")
        device_labels["cuda"] = f"CUDA ({torch.cuda.get_device_name(0)[:30]})"
    if torch.backends.mps.is_available():
        device_options.append("mps")
        device_labels["mps"] = "Apple MPS"
    # AMD ROCm 检测
    import os as _os
    if _os.environ.get("ROCM_HOME") or _os.environ.get("HIP_HOME"):
        device_options.append("hip")
        device_labels["hip"] = "AMD HIP (ROCm)"
    device_options.append("cpu")
    device_labels["cpu"] = "CPU"

    predict_device = st.selectbox(
        "预测设备", device_options,
        index=0,
        format_func=lambda x: device_labels.get(x, x),
        help="首次运行时生效；重启 app 后切换设备。",
    )
    if predict_device != "auto":
        _os.environ["MATGEN_DEVICE"] = predict_device

    st.divider()
    st.caption("MatGen-Eq v2.0 | 西南交通大学")


# ── 主界面标题 ───────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🧪 MatGen-Eq 智能材料发现系统")
    st.caption("候选结构生成 · ΔG_H 预测 · 专家审查 · 数据回流")
with col_status:
    st.markdown("")
    stats = db.get_stats()
    state_dist = db.get_state_distribution()
    total = stats.get("total", 0)
    st.metric("数据库样本", total)
    filtered_in_count = state_dist.get("filtered_in", 0)
    if filtered_in_count > 0:
        st.success(f"📋 {filtered_in_count} 条待审查")

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 结构生成（RAG）",
    "📊 上传预测（Eqv2-Lite）",
    "🔄 闭环批处理",
    "📋 专家审查台",
])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — RAG 生成 POSCAR
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("HEA-Gen 候选结构生成")

    # 提示词模板辅助
    with st.expander("💡 快速提示词模板", expanded=False):
        st.markdown("""
        **结构描述模板：**
        - `在 {element_list} 表面上吸附一个 H 原子，晶格常数约 {a} Å`
        - `生成 {element_list} 高熵合金表面结构，H覆盖率为 {coverage} ML`
        - `设计 {element_list} 合金的 {miller_index} 表面，原子数 20-50`
        """)
        if default_elements:
            st.code(f"在 {'/'.join(default_elements)} 表面上吸附一个 H 原子，给出完整的 VASP POSCAR 格式结构。", language="text")

    prompt = st.text_area(
        "结构描述提示词",
        placeholder=f"例如：在 {'/'.join(default_elements)} 表面上吸附一个 H 原子，给出合理的晶格与坐标。",
        height=180,
        key="tab1_prompt",
    )

    col_gen1, col_gen2, col_gen3 = st.columns(3)
    with col_gen1:
        max_new_tokens = st.number_input(
            "max_new_tokens", min_value=256, max_value=4096,
            value=max_new_tokens_side, step=128, key="tab1_tokens",
        )
    with col_gen2:
        temperature = st.number_input(
            "temperature", min_value=0.0, max_value=2.0,
            value=temperature_side, step=0.05, format="%.2f", key="tab1_temp",
        )
    with col_gen3:
        top_p = st.number_input(
            "top_p", min_value=0.0, max_value=1.0,
            value=top_p_side, step=0.05, format="%.2f", key="tab1_top_p",
        )

    col_mode1, col_mode2, col_mode3 = st.columns(3)
    with col_mode1:
        num_candidates = st.number_input(
            "生成候选数量",
            min_value=1, max_value=10, value=num_candidates_side, step=1,
            key="tab1_num_candidates",
        )
    with col_mode2:
        enable_diversity = st.checkbox("多样性采样", value=False, key="tab1_diversity")
    with col_mode3:
        enable_rag = st.checkbox("启用 RAG 检索增强", value=True, key="tab1_rag",
            help="从训练集检索相似参考结构，引导生成更准确")

    if st.button("🚀 生成候选结构", key="btn_generate", type="primary"):
        if not prompt.strip():
            st.error("请输入结构描述。")
        else:
            log_placeholder = st.empty()
            log_holder = ['<div class="log-container">']
            def update_log(level, msg):
                log_holder[0] += build_log_line(level, msg) + "<br>"
                log_placeholder.markdown(log_holder[0], unsafe_allow_html=True)

            update_log("info", f"开始生成 {num_candidates} 个候选（RAG={'开' if enable_rag else '关'}）...")
            with st.spinner("HEA-Gen 推理中，请稍候…"):
                try:
                    from backend.rag_gen import generate
                    results = []
                    actual_temp = 0.3 if (enable_diversity and int(num_candidates) > 1) else temperature
                    for cand_idx in range(int(num_candidates)):
                        cand_temp = actual_temp + cand_idx * 0.05
                        update_log("info", f"[{cand_idx+1}/{num_candidates}] temperature={cand_temp:.2f}")
                        r = generate(
                            prompt.strip(),
                            max_new_tokens=int(max_new_tokens),
                            temperature=cand_temp,
                            top_p=float(top_p),
                            base_model_name_or_path=BASE_MODEL_PATH,
                            use_rag=enable_rag,
                            rag_top_k=3,
                        )
                        results.append(r)
                        update_log("info", f"候选 {cand_idx+1} 生成完成，长度={len(r)} 字符")

                    st.session_state["generated_poscar"] = results[0]
                    st.session_state["all_candidates"] = results
                    update_log("info", f"✅ 全部 {len(results)} 个候选生成完毕")
                    st.success(f"生成完成（{len(results)} 个候选）")
                except ImportError as e:
                    update_log("error", f"模型依赖未安装：{e}")
                    st.error(f"模型依赖未安装：{e}")
                except FileNotFoundError as e:
                    update_log("error", f"模型文件未找到：{e}")
                    st.error(f"模型文件未找到：{e}")
                except Exception as e:
                    update_log("error", f"生成失败：{traceback.format_exc()}")
                    st.error(f"生成失败：{e}")

    st.divider()

    if st.session_state.get("generated_poscar"):
        st.subheader("📄 生成的 POSCAR（候选 1）")
        st.code(st.session_state["generated_poscar"], language="text")

        # RAG 检索可视化
        from backend.rag_visual import load_near_neighbors, get_reference_structures, format_rag_visualization
        import random
        nni = load_near_neighbors()
        if nni:
            random.seed(42)
            sample_idx = random.randint(0, len(nni) - 1)
            ref_idx = nni[sample_idx]
            ref_indices = [ref_idx]
            viz_text = format_rag_visualization(prompt, ref_indices)
            with st.expander("🔍 RAG 检索参考结构（本次生成）"):
                st.markdown(viz_text)
        else:
            st.info("RAG 参考结构索引不可用")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "💾 下载 POSCAR",
                data=st.session_state["generated_poscar"],
                file_name="generated.vasp",
                mime="text/plain",
                key="dl_poscar_tab1",
            )
        with col_dl2:
            if st.button("🔮 立即预测 ΔG_H", key="btn_predict_from_gen"):
                from backend.eq_predict import predict
                from backend.quality import validate_structure
                try:
                    pred = predict(st.session_state["generated_poscar"])
                    dg_h = pred["dg_h"]
                    st.session_state["last_prediction"] = pred
                    struct_check = validate_structure(st.session_state["generated_poscar"])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("ΔG_H (eV)", f"{dg_h:.4f}")
                    c2.metric("原子数", pred["num_atoms"])
                    c3.metric("设备", pred["device"])

                    deviation = abs(dg_h - target_dg_h)
                    if deviation <= tolerance:
                        st.success(f"✅ 偏差 {deviation:.4f} eV 在容差 ±{tolerance} eV 范围内")
                    else:
                        st.warning(f"⚠️ 偏差 {deviation:.4f} eV 超出容差范围")

                    if not struct_check["valid"]:
                        st.warning("⚠️ 结构异常：" + "; ".join(struct_check["warnings"]))
                    else:
                        st.info("✓ 结构校验通过")
                except Exception as e:
                    st.error(f"预测失败：{e}")

        # 多候选快速对比
        all_candidates = st.session_state.get("all_candidates", [])
        if len(all_candidates) > 1:
            st.divider()
            st.subheader(f"📑 多候选对比（共 {len(all_candidates)} 个）")
            for i, cand in enumerate(all_candidates, 1):
                with st.expander(f"候选 {i}"):
                    st.code(cand, language="text")

            if st.button("🔮 批量预测 ΔG_H（多候选）", key="btn_batch_predict"):
                from backend.eq_predict import predict
                pred_results = []
                for i, cand in enumerate(all_candidates, 1):
                    try:
                        pred = predict(cand)
                        deviation = abs(pred["dg_h"] - target_dg_h)
                        status = "✅" if deviation <= tolerance else "❌"
                        pred_results.append({
                            "idx": i,
                            "dg_h": pred["dg_h"],
                            "deviation": deviation,
                            "num_atoms": pred["num_atoms"],
                            "status": status,
                        })
                    except Exception as e:
                        pred_results.append({"idx": i, "dg_h": None, "error": str(e)})

                st.markdown("**多候选 ΔG_H 预测结果：**")
                rows = []
                for pr in pred_results:
                    if pr.get("dg_h") is not None:
                        rows.append({
                            "候选": f"候选 {pr['idx']}",
                            "ΔG_H (eV)": f"{pr['dg_h']:.4f}",
                            f"|ΔG_H - {target_dg_h}|": f"{pr['deviation']:.4f}",
                            "原子数": pr['num_atoms'],
                            "状态": pr['status'],
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — 上传 POSCAR 预测
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Eqv2-Lite ΔG_H 预测")

    use_generated = st.checkbox(
        "使用 Tab 1 生成的 POSCAR",
        value=bool(st.session_state.get("generated_poscar")),
        key="use_generated_t2",
    )

    poscar_text_t2 = None

    if use_generated:
        poscar_text_t2 = st.session_state.get("generated_poscar", "")
        if poscar_text_t2:
            with st.expander("📄 当前 POSCAR（来自 Tab 1）"):
                st.code(poscar_text_t2, language="text")
        else:
            st.warning("Tab 1 尚未生成 POSCAR。")
    else:
        uploaded = st.file_uploader(
            "上传 POSCAR / VASP 文件",
            type=["vasp", "poscar", "txt"],
            help="支持 VASP / POSCAR / TXT 格式",
            key="poscar_upload_t2",
        )
        if uploaded:
            poscar_text_t2 = uploaded.read().decode("utf-8", errors="replace")
            with st.expander("📄 已上传文件内容"):
                st.code(poscar_text_t2, language="text")

    if st.button("🔮 预测 ΔG_H", disabled=(not poscar_text_t2), key="btn_predict_t2", type="primary"):
        with st.spinner("Eqv2-Lite 推理中…"):
            try:
                from backend.eq_predict import predict
                from backend.quality import validate_structure

                res = predict(poscar_text_t2)
                dg_h = res["dg_h"]
                st.session_state["last_prediction"] = res
                deviation = abs(dg_h - target_dg_h)

                # 预测卡片
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ΔG_H (eV)", f"{dg_h:.4f}")
                m2.metric(f"|ΔG_H - {target_dg_h}|", f"{deviation:.4f}")
                m3.metric("原子数", res["num_atoms"])
                m4.metric("运行设备", res["device"])

                # HER 潜力评估
                if deviation <= tolerance:
                    st.success(f"✅ 候选结构偏差 {deviation:.4f} eV，在容差 ±{tolerance} eV 范围内，具有良好的 HER 催化潜力。")
                elif dg_h < target_dg_h - tolerance:
                    st.warning(f"⚠️ ΔG_H = {dg_h:.4f} eV，过负，氢吸附过强。")
                elif dg_h > target_dg_h + tolerance:
                    st.warning(f"⚠️ ΔG_H = {dg_h:.4f} eV，过正，氢吸附过弱。")

                # 结构校验
                struct_check = validate_structure(poscar_text_t2)
                if not struct_check["valid"]:
                    st.warning("⚠️ 结构异常：" + "；".join(struct_check["warnings"]))
                else:
                    st.info("✓ 结构校验通过")

                with st.expander("🔍 结构详情"):
                    info = struct_check.get("info", {})
                    col1, col2 = st.columns(2)
                    col1.write(f"原子数：{info.get('num_atoms', 'N/A')}")
                    col1.write(f"元素种类数：{info.get('num_elements', 'N/A')}")
                    col2.write(f"最小原子间距：{info.get('min_distance', 'N/A')} Å")
                    if struct_check["warnings"]:
                        for w in struct_check["warnings"]:
                            st.warning(w)

                # 保存到数据库
                in_filter = (abs(dg_h - target_dg_h) <= tolerance)
                rejection_level = struct_check.get("rejection_level", "pass") if struct_check.get("valid") else "structure"
                saved_id = db.save_sample(
                    prompt="[Tab2上传预测]",
                    poscar=poscar_text_t2,
                    dg_h=dg_h,
                    in_filter=in_filter,
                    rejection_level=rejection_level,
                    current_state="predicted",
                )
                # 状态转移
                if in_filter:
                    db.transition_state(saved_id, "predicted", "预测完成")
                    db.transition_state(saved_id, "filtered_in", f"ΔG_H={dg_h:.4f}在容差内")
                else:
                    db.transition_state(saved_id, "predicted", "预测完成")
                    db.transition_state(saved_id, "filtered_out", f"ΔG_H={dg_h:.4f}超出容差")
                st.info(f"💾 已保存到数据库，ID={saved_id}")

            except ImportError as e:
                st.error(f"Eqv2-Lite 依赖未安装：{e}")
            except ValueError as e:
                st.error(f"POSCAR 解析失败：{e}")
            except Exception as e:
                st.error(f"预测失败：{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — 闭环批处理
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("吸附氢能闭环批处理流水线")

    # 筛选参数展示
    col_f1, col_f2 = st.columns(2)
    col_f1.metric("筛选下限", f"{target_dg_h - tolerance:.2f} eV")
    col_f2.metric("筛选上限", f"{target_dg_h + tolerance:.2f} eV")

    st.divider()

    batch_prompts_raw = st.text_area(
        "📝 批量提示词（每行一条）",
        placeholder="在 Cu(111) 表面吸附一个 H 原子，给出完整的 VASP POSCAR 格式结构。\n生成 Ir/Pd/Pt 高熵合金表面结构，H覆盖率为 1/4 ML。",
        height=160,
        key="batch_prompts_t3",
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        filter_low = st.number_input("筛选下限 (eV)", value=target_dg_h - tolerance, step=0.1, format="%.2f", key="filter_low_t3")
    with bc2:
        filter_high = st.number_input("筛选上限 (eV)", value=target_dg_h + tolerance, step=0.1, format="%.2f", key="filter_high_t3")
    with bc3:
        st.write("")  # spacer
        run_batch = st.button("🚀 运行闭环批处理", key="btn_batch_t3", type="primary")

    if run_batch:
        prompts = [p.strip() for p in batch_prompts_raw.strip().split("\n") if p.strip()]
        if not prompts:
            st.error("请输入至少一条提示词。")
        else:
            from backend.rag_gen import generate as rag_generate
            from backend.eq_predict import predict as eq_predict

            # 日志区域
            log_container = st.empty()
            log_lines = []
            def add_log(level, msg):
                log_lines.append(build_log_line(level, msg))
                log_container.markdown('<div class="log-container">' + "<br>".join(log_lines) + "</div>", unsafe_allow_html=True)

            progress_bar = st.progress(0, text="初始化…")
            start_time = time.time()

            saved_ids = []
            for i, p in enumerate(prompts):
                progress_bar.progress((i + 1) / len(prompts), text=f"处理第 {i+1}/{len(prompts)} 条…")
                add_log("info", f"[{i+1}/{len(prompts)}] 生成中：{p[:50]}…")

                poscar = None
                dg_h = None
                in_filter = False
                rejection_level = "text"

                try:
                    poscar = rag_generate(p, base_model_name_or_path=BASE_MODEL_PATH)
                    add_log("info", f"  → POSCAR 生成成功，长度={len(poscar)} 字符")
                except Exception as e:
                    add_log("warn", f"  → 生成失败：{e}")

                if poscar:
                    try:
                        from backend.quality import validate_structure
                        check = validate_structure(poscar)
                        rejection_level = check.get("rejection_level", "pass")
                        if not check["valid"]:
                            add_log("warn", f"  → 结构校验警告：{'；'.join(check['warnings'])}")
                    except Exception as e:
                        rejection_level = "structure"
                        add_log("warn", f"  → 校验异常：{e}")
                else:
                    rejection_level = "text"

                if poscar and rejection_level in ("pass", ""):
                    try:
                        pred = eq_predict(poscar)
                        dg_h = pred["dg_h"]
                        in_filter = (float(filter_low) <= dg_h <= float(filter_high))
                        if in_filter:
                            add_log("info", f"  → ΔG_H={dg_h:.4f} ✅ 入围")
                        else:
                            add_log("warn", f"  → ΔG_H={dg_h:.4f} ❌ 淘汰")
                    except Exception as e:
                        add_log("warn", f"  → 预测失败：{e}")

                sample_id = db.save_sample(p, poscar or "", dg_h, in_filter, rejection_level)
                saved_ids.append(sample_id)

                # 状态转移
                if rejection_level == "pass" and poscar and dg_h is not None:
                    if in_filter:
                        db.transition_state(sample_id, "predicted", "预测完成")
                        db.transition_state(sample_id, "filtered_in", f"ΔG_H={dg_h:.4f}在容差内")
                    else:
                        db.transition_state(sample_id, "predicted", "预测完成")
                        db.transition_state(sample_id, "filtered_out", f"ΔG_H={dg_h:.4f}超出容差")
                elif poscar:
                    db.transition_state(sample_id, "rejected_precheck", f"rejection_level={rejection_level}")

            elapsed = time.time() - start_time
            add_log("info", f"✅ 批处理完成，共 {len(prompts)} 条，耗时 {elapsed:.1f}s，保存 {len(saved_ids)} 条到数据库")
            progress_bar.progress(1.0, text="完成")

            # 汇总统计
            st.divider()
            st.subheader("📊 本批次统计摘要")

            all_samples = db.get_all_samples()
            recent = all_samples[:len(prompts)]
            from backend.stats import compute_summary_stats, prepare_deviation_histogram
            from backend.cluster import cluster_by_dg_h, select_best_per_cluster, summarize_clusters

            summary = compute_summary_stats(recent, target_dg_h=target_dg_h)
            summary["avg_deviation"] = summary.get("avg_deviation") or 0
            summary["min_deviation"] = summary.get("min_deviation") or 0

            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("总生成", summary["total"])
            s2.metric("入围数", sum(1 for r in recent if r.get("in_filter")))
            s3.metric("通过率", f'{summary["pass_rate"]:.1%}')
            s4.metric("生成失败率", f'{summary["failure_rate"]:.1%}')
            s5.metric("平均|ΔG_H偏差|", f'{summary["avg_deviation"]:.4f}')
            s6.metric("最优|ΔG_H偏差|", f'{summary["min_deviation"]:.4f}')

            # ΔG_H 分布直方图
            from backend.db import get_distribution
            dist = get_distribution(bin_count=15)
            if dist["bins"]:
                st.divider()
                st.subheader("📈 ΔG_H 分布直方图")
                from backend.stats import prepare_histogram_data
                chart_data = prepare_histogram_data(dist)
                st.bar_chart(chart_data, x="bins", y="counts")
                st.caption(f"HER 理想区间 [{filter_low:.2f}, {filter_high:.2f}] eV，共 {dist['total']} 条历史样本")

            # 偏差直方图
            dev_hist = prepare_deviation_histogram(recent, target_dg_h=target_dg_h, bin_count=10)
            if dev_hist["bins"]:
                st.divider()
                st.subheader(f"📉 |ΔG_H - {target_dg_h}| 偏差分布")
                st.bar_chart({"偏差区间": dev_hist["bins"], "数量": dev_hist["counts"]}, x="偏差区间", y="数量")
                st.caption("越接近 0 表示热力学一致性越好")

            # 聚类分析
            clusters = cluster_by_dg_h(recent, n_clusters=3)
            summaries = summarize_clusters(clusters)
            best_per_cluster = select_best_per_cluster(clusters)

            with st.expander("🔗 聚类分析（按 ΔG_H 分组）"):
                for s in summaries:
                    st.write(
                        f"聚类 {s['cluster']}: 均值={s['mean']:.4f} eV, "
                        f"标准差={s['std']:.4f}, 计数={s['count']}, "
                        f"代表ID={s['representative_id']}, "
                        f"范围=[{s['dg_h_range'][0]:.4f}, {s['dg_h_range'][1]:.4f}]"
                    )
                st.write("**每个聚类最优代表：**")
                for b in best_per_cluster:
                    st.write(f"  ID={b.get('id')}: ΔG_H={b.get('dg_h'):.4f} eV")

    # 历史记录区
    st.divider()
    st.subheader("📜 历史记录")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        dg_h_low = st.number_input("ΔG_H 下限", value=-2.0, step=0.1, key="filter_dg_h_low_t3")
    with f2:
        dg_h_high = st.number_input("ΔG_H 上限", value=1.0, step=0.1, key="filter_dg_h_high_t3")
    with f3:
        h_status = st.selectbox("复核状态", ["all", "pending", "approved", "rejected"], key="filter_human_t3")
    with f4:
        v_status = st.selectbox("验证状态", ["all", "pending", "validated", "invalid"], key="filter_valid_t3")

    filtered_samples = db.query_samples(
        dg_h_min=float(dg_h_low),
        dg_h_max=float(dg_h_high),
        human_status=h_status,
        validation_status=v_status,
        limit=200,
    )
    st.caption(f"筛选结果：{len(filtered_samples)} 条（最多 200 条）")
    show_batch_table(filtered_samples, show_uuid=True, target_dg_h=target_dg_h)

    # 全量历史
    if filtered_samples:
        st.divider()
        all_samples = db.get_all_samples()
        st.subheader(f"全部历史记录（共 {len(all_samples)} 条）")
        show_batch_table(all_samples[:100], show_uuid=True, target_dg_h=target_dg_h)

        # POSCAR 详情
        with st.expander("🔍 POSCAR 详情（前 20 条）"):
            for s in all_samples[:20]:
                dg_str = f"{s['dg_h']:.4f}" if s.get("dg_h") else "—"
                with st.expander(f"ID {s['id']} | ΔG_H={dg_str} | {s.get('prompt','')[:40]}"):
                    st.code(s.get("poscar", "(empty)"), language="text")

        # 候选质量排序
        st.divider()
        st.subheader(f"🏆 候选质量排序（按 |ΔG_H - {target_dg_h}| 升序）")
        from backend.quality import sort_candidates_by_quality
        sorted_samples = sort_candidates_by_quality(all_samples[:100], target_dg_h=target_dg_h)
        show_batch_table(sorted_samples, show_uuid=True, target_dg_h=target_dg_h)
        st.caption(f"排序基准：HER 理想 ΔG_H ≈ {target_dg_h} eV，越接近越优先")

    # 统计面板
    st.divider()
    st.subheader("📊 数据库统计面板")

    with st.expander("统计详情"):
        stats = db.get_stats()
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("总样本", stats.get("total", 0))
        s2.metric("待复核", stats.get("pending", 0))
        s3.metric("已批准", stats.get("approved", 0))
        s4.metric("已拒绝", stats.get("rejected", 0))
        s5.metric("已验证", stats.get("validated", 0))

        # 状态分布
        state_dist = db.get_state_distribution()
        if state_dist:
            st.markdown("**状态分布：**")
            st_data = [{"状态": k, "数量": v} for k, v in state_dist.items()]
            st.dataframe(pd.DataFrame(st_data), use_container_width=True, hide_index=True)

        # ΔG_H 分布
        from backend.db import get_distribution
        dist = get_distribution(bin_count=15)
        summary = compute_summary_stats(db.get_all_samples(), target_dg_h=target_dg_h)
        if dist["bins"]:
            st.bar_chart(prepare_histogram_data(dist), x="bins", y="counts")

        # rejection_level 分布
        from backend.db import get_rejection_level_stats
        rl_stats = get_rejection_level_stats()
        total_rl = rl_stats.pop("_total", 0)
        if rl_stats:
            rl_data = {"rejection_level": list(rl_stats.keys()), "count": list(rl_stats.values())}
            st.bar_chart(rl_data, x="rejection_level", y="count")
            st.caption(f"总计 {total_rl} 条样本")

    # 结构对比
    st.divider()
    st.subheader("⚖️ 候选结构横向对比")
    cmp1, cmp2, cmp3 = st.columns(3)
    with cmp1:
        id_a = st.text_input("候选 A ID", value="", key="cmp_id_a_t3", placeholder="输入样本ID")
    with cmp2:
        id_b = st.text_input("候选 B ID", value="", key="cmp_id_b_t3", placeholder="输入样本ID")
    with cmp3:
        st.write("")  # spacer
        if st.button("⚖️ 对比两个候选", key="btn_compare_t3"):
            if not id_a.strip() or not id_b.strip():
                st.error("请输入两个候选 ID")
            else:
                from backend.compare import compare_two
                from backend.db import get_sample
                s_a = get_sample(int(id_a.strip()))
                s_b = get_sample(int(id_b.strip()))
                if not s_a or not s_b:
                    st.error("未找到对应 ID 的样本")
                elif not s_a.get("poscar") or not s_b.get("poscar"):
                    st.error("样本无 POSCAR 数据")
                else:
                    result = compare_two(s_a["poscar"], s_b["poscar"])
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        colA, colB = st.columns(2)
                        with colA:
                            st.write(f"**候选 A（ID={id_a.strip()}）**")
                            st.json({k: v for k, v in result["a_info"].items() if k != "elements"})
                            st.write(f"ΔG_H: {s_a.get('dg_h', 'N/A'):.4f} eV" if s_a.get("dg_h") else "ΔG_H: N/A")
                        with colB:
                            st.write(f"**候选 B（ID={id_b.strip()}）**")
                            st.json({k: v for k, v in result["b_info"].items() if k != "elements"})
                            st.write(f"ΔG_H: {s_b.get('dg_h', 'N/A'):.4f} eV" if s_b.get("dg_h") else "ΔG_H: N/A")
                        st.divider()
                        st.write("**差异（B - A）：**")
                        st.json(result["diff"])
                        st.caption("正值表示 B 的该参数大于 A")


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — 专家审查台
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("👔 专家审查与数据回流")

    st.markdown("""
    <div class="card">
    <div class="card-header">审查说明</div>
    下方展示所有 <code>filtered_in</code> 状态的候选结构。请逐一审查，综合考虑原子间距合理性、
    元素分布及 HER 催化潜力，做出通过或驳回决策。确认通过的结构将进入训练数据回流队列。
    </div>
    """, unsafe_allow_html=True)

    # 获取待审查样本
    filtered_in_samples = db.get_filtered_in_samples(limit=100)

    if not filtered_in_samples:
        st.info("暂无待审查的候选结构（filtered_in 状态）。请先在 Tab 3 运行批处理。")
    else:
        st.success(f"📋 当前共有 {len(filtered_in_samples)} 条待审查候选")

        # 批量操作
        col_review1, col_review2, col_review3 = st.columns(3)
        with col_review1:
            select_all = st.button("☑️ 全选", key="select_all_review")
        with col_review2:
            batch_validate = st.button("✅ 批量确认通过", key="batch_validate")
        with col_review3:
            batch_reject = st.button("❌ 批量驳回", key="batch_reject")

        st.divider()

        # 筛选辅助
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            deviation_max = st.number_input(
                f"显示 |ΔG_H - {target_dg_h}| ≤ 的值",
                value=0.5, min_value=0.0, step=0.05, format="%.2f",
                key="review_deviation_filter",
            )
        with col_filter2:
            sort_by = st.selectbox(
                "排序方式",
                ["deviation_asc", "dg_h_asc", "dg_h_desc", "id_desc"],
                key="review_sort_by",
            )

        # 过滤并排序
        filtered = []
        for s in filtered_in_samples:
            if s.get("dg_h") is not None:
                dev = abs(s["dg_h"] - target_dg_h)
                if dev <= deviation_max:
                    s["_deviation"] = dev
                    filtered.append(s)

        if sort_by == "deviation_asc":
            filtered.sort(key=lambda x: x["_deviation"])
        elif sort_by == "dg_h_asc":
            filtered.sort(key=lambda x: x.get("dg_h") or 999)
        elif sort_by == "dg_h_desc":
            filtered.sort(key=lambda x: x.get("dg_h") or -999, reverse=True)
        else:
            filtered.sort(key=lambda x: x.get("id") or 0, reverse=True)

        st.caption(f"筛选后：{len(filtered)} 条")

        # 卡片式展示
        for s in filtered:
            dev = s.get("_deviation", 0)
            dg_h = s.get("dg_h")
            uuid_short = make_uuid_short(s.get("uuid", ""))

            card_class = "review-card-filtered"
            with st.container():
                st.markdown(f"""
                <div class="review-card {card_class}">
                    <table style="width:100%; border-collapse:collapse;">
                        <tr>
                            <td style="padding:0.3rem; font-weight:bold; color:#1e3a5f; width:80px;">ID</td>
                            <td style="padding:0.3rem;">{s['id']} <span style="font-size:0.75rem; color:#888;">({uuid_short})</span></td>
                            <td style="padding:0.3rem; font-weight:bold; color:#1e3a5f; width:100px;">ΔG_H</td>
                            <td style="padding:0.3rem; font-size:1.1rem; font-weight:800; color:{'#28a745' if dev <= tolerance else '#e67e22'};">{dg_h:.4f} eV</td>
                            <td style="padding:0.3rem; font-weight:bold; color:#1e3a5f; width:120px;">|偏差|</td>
                            <td style="padding:0.3rem; color:{'#28a745' if dev <= tolerance else '#e67e22'};">{dev:.4f} eV</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                # 提示词预览
                st.markdown(f"**提示词：** `{s.get('prompt', '')[:80]}…`")

                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
                with col_btn1:
                    if st.button(f"✅ 通过", key=f"approve_{s['id']}"):
                        db.update_review(s["id"], human_status="approved", validation_status="validated")
                        db.transition_state(s["id"], "validated", "专家确认通过")
                        st.rerun()
                with col_btn2:
                    if st.button(f"❌ 驳回", key=f"reject_{s['id']}"):
                        db.update_review(s["id"], human_status="rejected", validation_status="invalid")
                        db.transition_state(s["id"], "rejected", "专家驳回")
                        st.rerun()
                with col_btn3:
                    with st.expander("📄 查看完整 POSCAR"):
                        st.code(s.get("poscar", "(empty)"), language="text")

                        # 提取元素信息
                        try:
                            import ase.io, io as _io
                            atoms = ase.io.read(_io.StringIO(s.get("poscar", "")), format="vasp")
                            elem_counts = atoms.get_chemical_formula(mode="count")
                            st.info(f"元素组成：{elem_counts}，原子数：{len(atoms)}")
                        except Exception:
                            pass

                st.divider()

        # 导出功能
        st.divider()
        st.subheader("💾 数据导出")

        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        with col_exp1:
            export_validated_json = st.button("📥 导出 validated (JSON)", key="export_validated_json")
        with col_exp2:
            export_all_json = st.button("📥 导出全部 (JSON)", key="export_all_json")
        with col_exp3:
            export_poscar_zip = st.button("📦 批量导出 POSCAR (ZIP)", key="export_poscar_zip")
        with col_exp4:
            export_filtered_zip = st.button("📦 导出 filtered_in POSCAR", key="export_filtered_zip")

        if export_validated_json:
            validated = db.export_validated()
            if validated:
                export_data = json.dumps(validated, ensure_ascii=False, indent=2)
                st.download_button(
                    f"💾 下载 {len(validated)} 条 validated 样本 (JSON)",
                    data=export_data,
                    file_name="validated_training_samples.json",
                    mime="application/json",
                    key="dl_validated_t4",
                )
            else:
                st.warning("暂无 validated 样本。")

        if export_all_json:
            all_s = db.get_all_samples()
            if all_s:
                export_data = json.dumps(all_s, ensure_ascii=False, indent=2)
                st.download_button(
                    f"💾 下载全部 {len(all_s)} 条样本 (JSON)",
                    data=export_data,
                    file_name="all_samples.json",
                    mime="application/json",
                    key="dl_all_t4",
                )
            else:
                st.warning("数据库为空。")

        # POSCAR ZIP 导出
        def generate_poscar_zip(samples: list, label: str) -> bytes:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for s in samples:
                    poscar = s.get("poscar", "")
                    if poscar:
                        filename = f"{label}_id{s['id']}_uuid{make_uuid_short(s.get('uuid', ''))}.vasp"
                        zf.writestr(filename, poscar)
            return zip_buffer.getvalue()

        if export_poscar_zip:
            all_s = db.get_all_samples()
            poscar_samples = [s for s in all_s if s.get("poscar")]
            if poscar_samp := [s for s in all_s if s.get("poscar")]:
                zip_data = generate_poscar_zip(poscar_samples, "all")
                st.download_button(
                    f"💾 下载 {len(poscar_samples)} 个 POSCAR (ZIP)",
                    data=zip_data,
                    file_name="all_poscars.zip",
                    mime="application/zip",
                    key="dl_poscar_zip_t4",
                )
            else:
                st.warning("没有可导出的 POSCAR。")

        if export_filtered_zip:
            if filtered_in_samples:
                zip_data = generate_poscar_zip(filtered_in_samples, "filtered_in")
                st.download_button(
                    f"💾 下载 {len(filtered_in_samples)} 个 filtered_in POSCAR (ZIP)",
                    data=zip_data,
                    file_name="filtered_in_poscars.zip",
                    mime="application/zip",
                    key="dl_filtered_zip_t4",
                )
            else:
                st.warning("没有 filtered_in 样本可导出。")

        # 状态流转历史
        st.divider()
        st.subheader("🔎 状态流转历史查询")
        col_hist1, col_hist2 = st.columns([1, 3])
        with col_hist1:
            hist_id = st.text_input("样本 ID", value="", key="hist_sample_id", placeholder="输入样本ID")
        with col_hist2:
            st.write("")
            if st.button("🔍 查询状态历史", key="btn_state_history_t4"):
                if hist_id.strip():
                    try:
                        sid = int(hist_id.strip())
                        history = db.get_state_history(sid)
                        if history:
                            st.write(f"**样本 {sid} 的状态流转历史：**")
                            rows = []
                            for h in history:
                                rows.append({
                                    "时间": h["changed_at"][:19],
                                    "从状态": h["from_state"] or "⊗",
                                    "到状态": h["to_state"],
                                    "原因": h["reason"],
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.info(f"暂无状态历史记录（样本 {sid}）")
                    except ValueError:
                        st.error("sample_id 必须是整数。")
                else:
                    st.warning("请输入 sample_id")

    # 复核状态更新（通用）
    st.divider()
    st.subheader("🔧 复核状态手动更新")
    col_up1, col_up2, col_up3, col_up4 = st.columns(4)
    with col_up1:
        sample_id_input = st.text_input("sample_id", key="rev_id_t4")
    with col_up2:
        human_status = st.selectbox("human_status", ["pending", "approved", "rejected"], key="rev_human_t4")
    with col_up3:
        validation_status = st.selectbox("validation_status", ["pending", "validated", "invalid"], key="rev_valid_t4")
    with col_up4:
        st.write("")
        if st.button("🔄 更新复核状态", key="btn_update_review_t4"):
            if not sample_id_input.strip():
                st.error("请输入 sample_id。")
            else:
                try:
                    sid = int(sample_id_input.strip())
                    ok = db.update_review(sid, human_status, validation_status)
                    if ok:
                        st.success(f"ID {sid} → human={human_status}，validation={validation_status}")
                        st.rerun()
                    else:
                        st.error(f"未找到 ID {sid}")
                except ValueError:
                    st.error("sample_id 必须是整数。")

    # 数据库统计
    st.divider()
    st.subheader("📊 数据库概览")
    stats = db.get_stats()
    state_dist = db.get_state_distribution()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总样本", stats.get("total", 0))
    c2.metric("待审查", state_dist.get("filtered_in", 0))
    c3.metric("已确认", state_dist.get("validated", 0))
    c4.metric("已驳回", state_dist.get("rejected", 0))
    c5.metric("已回流", state_dist.get("exported_for_training", 0))
