# app.py
import sys
import os
import streamlit as st
import requests
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from labels import state_label, localize_record, localize_distribution

API_BASE = os.environ.get("MATGEN_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MatGen-Eq 候选结构生成与筛选系统",
    page_icon="🔬",
    layout="wide"
)

st.title("MatGen-Eq 候选结构生成与筛选原型系统")
st.markdown("**四层微服务架构**：交互层 → 任务编排层 → 算法适配层 → 数据持久化层")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 结构生成", "📊 预测结果", "🔍 专家审查",
    "⚗️ DFT 回填", "🏗️ 架构监控", "🔄 飞轮全景",
])

with tab1:
    st.header("结构生成参数配置")
    col1, col2 = st.columns(2)
    with col1:
        target_dgH = st.number_input("目标 ΔG_H (eV)", value=-0.50, step=0.01, format="%.2f")
        tolerance = st.number_input("容差范围 (eV)", value=0.05, step=0.01, format="%.2f")
        batch_size = st.slider("生成数量", 5, 100, 10)
    with col2:
        st.markdown("**逆向设计输入说明**")
        st.info(
            "本系统以目标性质（ΔG_H）为输入进行逆向设计，"
            "**元素种类与配比由模型自主生成**，无需也不应由用户指定。"
        )

    if st.button("🚀 开始生成", type="primary"):
        with st.spinner("正在连接任务编排层..."):
            try:
                response = requests.post(
                    f"{API_BASE}/generate",
                    json={
                        "target_dgH": target_dgH,
                        "tolerance": tolerance,
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
                        with st.expander(f"结构 {r['uuid'][:8]}... - {state_label(r['status'])}"):
                            st.json(localize_record(r))
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
        "⚠️ 专家审查前须先在【⚗️ DFT 回填】完成 DFT 录入，结构状态需为「DFT已验证」(dft_verified)。"
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
        "结构状态将从 **初筛通过**(filtered_in) 推进至 **DFT已验证**(dft_verified)。"
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
                    st.success(f"回填成功，状态已更新为「DFT已验证」（UUID: {dft_uuid[:8]}...）")
                elif resp.status_code == 404:
                    st.error("结构不存在，请检查 UUID")
                elif resp.status_code == 409:
                    st.error("状态转移失败：结构当前状态不是「初筛通过」(filtered_in)")
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

with tab6:
    st.header("🔄 飞轮全景监控")

    # ── 飞轮 8 环节流程图（文字版，答辩讲解用）────────────────────────────
    st.subheader("飞轮 8 环节闭环")
    st.markdown("""
    ```
    ①生成 → ②预测 → ③初筛 → ④DFT回填 → ⑤专家确认 → ⑥导出 → ⑦数据集回灌 → ⑧重训+版本注册
       ↑                                                                              │
       └──────────────────────── 飞轮闭环 ─────────────────────────────────────────────┘
    ```
    | 环节 | 动作 | 输出状态 |
    |------|------|----------|
    | ① 生成 | HEA-Gen / fake 生成 POSCAR | `generated` |
    | ② 预测 | Eqv2-Lite / toy_mlp 预测 ΔG_H | `predicted` |
    | ③ 初筛 | ΔG_H 容差过滤 | `filtered_in` / `filtered_out` |
    | ④ DFT 回填 | 外部 DFT 结果写入 | `dft_verified` |
    | ⑤ 专家确认 | 人工审查质量 | `validated` / `rejected` |
    | ⑥ 导出 | export_validated() 取数据 | — |
    | ⑦ 数据集回灌 | 追加训练集 + 重建近邻索引 | `exported_for_training` |
    | ⑧ 重训注册 | demo/真实 retrain + 版本号 | 新版本 v{n} |
    """)

    st.divider()

    # ── 各状态结构计数 ─────────────────────────────────────────────────────
    st.subheader("各状态结构计数")
    try:
        resp_states = requests.get(f"{API_BASE}/stats/states", timeout=5)
        if resp_states.status_code == 200:
            dist = resp_states.json()
            if dist:
                # st.metric 展示
                state_cols = st.columns(min(len(dist), 5))
                for i, (state, count) in enumerate(sorted(dist.items())):
                    state_cols[i % len(state_cols)].metric(state_label(state), count)
                # 柱状图（中文状态标签）
                st.bar_chart(localize_distribution(dist))
            else:
                st.info("数据库暂无结构记录（尚未执行生成任务）")
        else:
            st.error(f"获取状态分布失败: {resp_states.status_code}")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API 服务未启动，请先运行: `python main.py`")
    except Exception as e:
        st.warning(f"⚠️ 获取状态分布失败: {e}")

    st.divider()

    # ── 模型版本历史 ───────────────────────────────────────────────────────
    st.subheader("模型版本历史")
    try:
        resp_ver = requests.get(f"{API_BASE}/models/versions", timeout=5)
        if resp_ver.status_code == 200:
            versions_data = resp_ver.json()
            if versions_data:
                for model_id, versions in versions_data.items():
                    with st.expander(f"模型: {model_id}  （共 {len(versions)} 个版本）"):
                        for v in reversed(versions):
                            st.markdown(
                                f"**{v.get('version','?')}**  |  "
                                f"{v.get('created_at','')[:19]}  |  "
                                f"_{v.get('notes','')}_"
                            )
                            if v.get("metrics"):
                                st.json(v["metrics"])
            else:
                st.info("暂无版本记录（尚未执行重训）")
        else:
            st.error(f"获取版本历史失败: {resp_ver.status_code}")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API 服务未启动，请先运行: `python main.py`")
    except Exception as e:
        st.warning(f"⚠️ 获取版本历史失败: {e}")

    st.divider()

    # ── 可用模型清单 ───────────────────────────────────────────────────────
    st.subheader("可用模型清单")
    try:
        resp_models = requests.get(f"{API_BASE}/models", timeout=5)
        if resp_models.status_code == 200:
            models_data = resp_models.json()
            col_gen, col_pred = st.columns(2)
            with col_gen:
                st.markdown("**生成模型**")
                for m in models_data.get("gen_models", []):
                    default_tag = " ★" if m["id"] == models_data.get("default_gen_model_id") else ""
                    st.markdown(f"- `{m['id']}`{default_tag}  {m.get('description','')}")
            with col_pred:
                st.markdown("**预测模型**")
                for m in models_data.get("pred_models", []):
                    default_tag = " ★" if m["id"] == models_data.get("default_pred_model_id") else ""
                    st.markdown(f"- `{m['id']}`{default_tag}  {m.get('description','')}")
        else:
            st.error(f"获取模型清单失败: {resp_models.status_code}")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API 服务未启动，请先运行: `python main.py`")
    except Exception as e:
        st.warning(f"⚠️ 获取模型清单失败: {e}")

    st.divider()

    # ── 一键 Demo 重训 ────────────────────────────────────────────────────
    st.subheader("一键 Demo 重训")
    retrain_kind = st.selectbox("选择重训模型类型", ["prediction", "generation"], key="flywheel_retrain_kind")
    if st.button("🚀 触发 Demo 重训", type="primary", key="flywheel_retrain_btn"):
        with st.spinner("正在执行 demo 重训（同步）..."):
            try:
                resp_retrain = requests.post(
                    f"{API_BASE}/models/retrain",
                    json={"model_kind": retrain_kind},
                    timeout=30,
                )
                if resp_retrain.status_code == 200:
                    retrain_result = resp_retrain.json()
                    nv = retrain_result.get("new_version", {})
                    st.success(
                        f"重训完成！新版本: **{nv.get('version','?')}**  |  "
                        f"注释: {nv.get('notes','')}"
                    )
                    fb = retrain_result.get("feedback", {})
                    st.markdown(
                        f"回灌统计: validated={fb.get('validated_count',0)}  "
                        f"appended={fb.get('appended',0)}  total={fb.get('total',0)}"
                    )
                    st.json(retrain_result)
                else:
                    st.error(f"重训失败: {resp_retrain.status_code} — {resp_retrain.text}")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ API 服务未启动，请先运行: `python main.py`")
            except Exception as e:
                st.error(f"重训出错: {e}")

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
    except Exception:
        st.warning("⚠️ API服务未响应，请运行 `python main.py` 启动服务")
