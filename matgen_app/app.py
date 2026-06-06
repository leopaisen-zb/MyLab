# app.py
import sys
import os
import streamlit as st
import requests
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

API_BASE = os.environ.get("MATGEN_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MatGen-Eq 材料发现系统",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 MatGen-Eq 智能材料发现系统")
st.markdown("**四层微服务架构**：交互层 → 任务编排层 → 算法适配层 → 数据持久化层")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 结构生成", "📊 预测结果", "🔍 专家审查", "⚗️ DFT 回填", "🏗️ 架构监控"])

with tab1:
    st.header("结构生成参数配置")
    col1, col2 = st.columns(2)
    with col1:
        target_dgH = st.number_input("目标 ΔG_H (eV)", value=-0.50, step=0.01, format="%.2f")
        tolerance = st.number_input("容差范围 (eV)", value=0.05, step=0.01, format="%.2f")
        batch_size = st.slider("生成数量", 5, 100, 10)
    with col2:
        elements = st.multiselect(
            "元素种类",
            options=["Ir", "Pd", "Pt", "Rh", "Ru", "Fe", "Co", "Ni"],
            default=["Ir", "Pd", "Pt", "Rh", "Ru"]
        )

    if st.button("🚀 开始生成", type="primary"):
        with st.spinner("正在连接任务编排层..."):
            try:
                response = requests.post(
                    f"{API_BASE}/generate",
                    json={
                        "target_dgH": target_dgH,
                        "tolerance": tolerance,
                        "elements": elements,
                        "batch_size": batch_size
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    task_id = data["task_id"]
                    st.success(f"任务已创建: {task_id}")
                    st.session_state["current_task_id"] = task_id
                    try:
                        requests.post(f"{API_BASE}/tasks/{task_id}/execute", timeout=10)
                    except Exception:
                        pass
                    st.info("任务已提交执行，点击下方【预测结果】标签页查看进度")
                else:
                    st.error(f"API错误: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ API服务未启动，请先运行: `python main.py`")

with tab2:
    st.header("📊 预测结果与筛选")
    if "current_task_id" in st.session_state:
        task_id = st.session_state["current_task_id"]
        if st.button("🔄 刷新状态"):
            st.rerun()
        try:
            resp = requests.get(f"{API_BASE}/tasks/{task_id}/status", timeout=10)
            if resp.status_code == 200:
                status = resp.json()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总数量", status["total"])
                col2.metric("已处理", status["processed"])
                col3.metric("筛选通过", status["success"])
                col4.metric("失败", status["failed"])

                if status["results"]:
                    st.subheader("结构详情")
                    for r in status["results"][:10]:
                        with st.expander(f"Structure {r['uuid'][:8]}... - {r['status']}"):
                            st.json(r)
            else:
                st.error(f"获取状态失败: {resp.status_code}")
        except Exception as e:
            st.info(f"请先生成结构，或检查API服务是否运行")
    else:
        st.info("请先在【结构生成】标签页创建生成任务")

with tab3:
    st.header("🔍 专家审查")
    st.info(
        "专家审查功能需要在任务完成且有筛选通过的结构后可用。"
        "⚠️ 专家审查前须先在【⚗️ DFT 回填】完成 DFT 录入，结构状态需为 dft_verified。"
    )
    structure_id = st.text_input("结构 UUID")
    col1, col2 = st.columns(2)
    if col1.button("✅ 确认为高质量结构"):
        if structure_id:
            resp = requests.post(f"{API_BASE}/structures/{structure_id}/validate?decision=validated")
            if resp.status_code == 200:
                st.success("结构已确认为高质量")
            elif resp.status_code == 409:
                st.warning("该结构尚未完成 DFT 验证，请先在 DFT 回填页录入")
            else:
                st.error(f"确认失败: {resp.status_code}")
    if col2.button("❌ 驳回"):
        if structure_id:
            resp = requests.post(f"{API_BASE}/structures/{structure_id}/validate?decision=rejected")
            if resp.status_code == 200:
                st.success("结构已驳回")
            elif resp.status_code == 409:
                st.warning("该结构尚未完成 DFT 验证，请先在 DFT 回填页录入")
            else:
                st.error(f"驳回失败: {resp.status_code}")

with tab4:
    st.header("⚗️ DFT 结果回填")
    st.info(
        "将外部 DFT 计算结果（ΔG_H / 总能量）回填入系统，"
        "结构状态将从 **filtered_in** 推进至 **dft_verified**。"
    )

    st.subheader("单条回填")
    dft_uuid = st.text_input("结构 UUID", key="dft_uuid")
    dft_dg_h = st.number_input("DFT ΔG_H (eV)", value=-0.48, step=0.01, format="%.4f", key="dft_dg_h")
    dft_energy = st.number_input(
        "DFT 总能量 (eV，可选，留 0 表示不填)", value=0.0, step=0.1, format="%.4f", key="dft_energy"
    )

    if st.button("提交 DFT 结果", type="primary", key="dft_submit"):
        if not dft_uuid:
            st.warning("请输入结构 UUID")
        else:
            payload = {"dft_dg_h": dft_dg_h}
            if dft_energy != 0.0:
                payload["dft_energy"] = dft_energy
            try:
                resp = requests.post(
                    f"{API_BASE}/structures/{dft_uuid}/dft",
                    json=payload,
                    timeout=10
                )
                if resp.status_code == 200:
                    st.success(f"回填成功，状态已更新为 dft_verified（UUID: {dft_uuid[:8]}...）")
                elif resp.status_code == 404:
                    st.error("结构不存在，请检查 UUID")
                elif resp.status_code == 409:
                    st.error("状态转移失败：结构当前状态不是 filtered_in")
                else:
                    st.error(f"API 错误: {resp.status_code} — {resp.text}")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ API 服务未启动，请先运行: `python main.py`")

    st.divider()
    st.subheader("批量回填（JSON）")
    st.markdown(
        "每行一条 JSON，或粘贴 JSON 数组。格式：`[{\"uuid\": \"...\", \"dft_dg_h\": -0.48}, ...]`"
    )
    batch_json = st.text_area("JSON 数组", height=150, key="dft_batch_json",
                              placeholder='[{"uuid": "abc-123", "dft_dg_h": -0.48}]')
    if st.button("批量提交", key="dft_batch_submit"):
        if not batch_json.strip():
            st.warning("请输入 JSON 数组")
        else:
            try:
                import json as _json
                items = _json.loads(batch_json)
                resp = requests.post(
                    f"{API_BASE}/structures/dft/batch",
                    json=items,
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"批量回填完成：成功 {data['success_count']} 条")
                    if data["failed"]:
                        st.warning(f"失败 UUID：{data['failed']}")
                else:
                    st.error(f"API 错误: {resp.status_code}")
            except ValueError as e:
                st.error(f"JSON 格式错误: {e}")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ API 服务未启动")

with tab5:
    st.header("🏗️ 系统架构监控")
    st.subheader("四层架构状态")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("交互层", "FastAPI", "运行中")
    col2.metric("任务编排层", "Orchestrator", "运行中")
    col3.metric("算法适配层", "Adapters", "就绪")
    col4.metric("数据持久化", "SQLite", "运行中")

    st.subheader("架构说明")
    st.markdown("""
    | 层次 | 组件 | 职责 |
    |------|------|------|
    | 交互层 | FastAPI + Streamlit | 接收HTTP请求，Web界面 |
    | 任务编排层 | TaskOrchestrator | 批量调度，状态机流转，检查点管理 |
    | 算法适配层 | HEAGenAdapter + EQAdapter | 封装异构模型，统一接口 |
    | 数据持久化层 | SQLite + Workspace | 结构化存储，文件管理 |
    """)
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            st.success("✅ API服务健康")
            st.json(resp.json())
    except:
        st.warning("⚠️ API服务未响应，请运行 `python main.py` 启动服务")
