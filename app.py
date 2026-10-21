"""Streamlit UI for local, consent-based biometric enrollment and comparison."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from faceauth.config import DEFAULT_DB, DEFAULT_THRESHOLD, MODEL_DIR, DETECTOR_FILE, RECOGNIZER_FILE
from faceauth.service import FaceAuthService
from faceauth.store import FaceStore
from faceauth.vision import FaceEngine, FaceInputError, read_photo

st.set_page_config(page_title="FaceAuth | 人脸身份认证实验系统", page_icon="🔐", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1240px; padding-top:1.55rem;}
[data-testid="stSidebar"] {background:#f3f6fa;}
.hero {background:linear-gradient(110deg,#142b40,#16666c); border-radius:18px;
    padding:26px 32px; margin-bottom:16px; color:#fff;}
.hero h1 {color:#fff; font-size:2rem; margin:0 0 8px;}
.hero p {color:#d9eef1; margin:0;}
.disclaimer {color:#546478; font-size:.92rem; margin-top:9px;}
.pill {display:inline-block; padding:5px 10px; background:#eaf4f1;
    color:#176b57; border-radius:20px; font-size:.82rem;}
</style>""", unsafe_allow_html=True)


def _photo_widget(key: str):
    source = st.radio("输入方式", ["上传图片", "摄像头拍照"], horizontal=True, key=f"{key}_source")
    if source == "上传图片":
        return st.file_uploader("选择照片（JPG、PNG、WebP）", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_upload")
    return st.camera_input("拍摄单人正脸照片", key=f"{key}_camera")


def _image_from_widget(widget):
    if widget is None:
        raise FaceInputError("请先上传照片或拍照。")
    image = read_photo(widget.getvalue())
    return image


@st.cache_resource(show_spinner="正在载入 YuNet / SFace 深度学习模型...")
def _get_engine() -> FaceEngine:
    return FaceEngine(MODEL_DIR, score_threshold=0.8, min_face_size=55)


@st.cache_resource(show_spinner=False)
def _get_store() -> FaceStore:
    return FaceStore(DEFAULT_DB, model_id=_get_engine().model_id)


st.markdown("""
<div class="hero">
<span style="font-size:0.77rem;letter-spacing:0.1em;color:#8de0d7">FACE AUTHENTICATION · LOCAL DEMO V1.6</span>
<h1>基于深度学习的人脸识别与身份认证系统</h1>
<p>YuNet 人脸检测 · SFace 特征提取 · 1:1 身份验证 · 1:N 身份检索 · 本地记录管理</p>
</div>
""", unsafe_allow_html=True)

model_ok = all((MODEL_DIR / filename).exists() for filename in (DETECTOR_FILE, RECOGNIZER_FILE))
if not model_ok:
    st.error("尚未安装预训练模型。先在 VSCode 终端运行 `python scripts/download_models.py`，完成后刷新页面。")
    st.code("python scripts/download_models.py\npython scripts/verify_env.py\nstreamlit run app.py --server.fileWatcherType none", language="powershell")
    st.caption("模型从 OpenCV 官方 Hugging Face / OpenCV Zoo 下载，并逐一进行 SHA-256 校验。")
    st.stop()

try:
    engine = _get_engine()
    store = _get_store()
    service = FaceAuthService(engine, store)
except Exception as exc:
    st.error(f"模型或本地数据库初始化失败：{exc}")
    st.stop()

with st.sidebar:
    st.header("系统设置")
    st.success("模型已加载 | 本地推理")
    persons = store.persons()
    st.metric("已注册人员", len(persons))
    st.metric("已注册人脸模板", store.sample_count())
    threshold = st.slider(
        "余弦相似度阈值", min_value=0.20, max_value=1.0,
        value=float(DEFAULT_THRESHOLD), step=0.01,
        help="越高通常越严格。当前默认值是演示起点，未经本项目数据校准，不能据此保证误识率。",
    )
    st.caption("示例阈值 0.45。该分数不是身份置信概率，真实系统应使用独立验证集校准 FAR/FRR。")
    st.divider()
    st.info("本地演示：默认仅存储特征向量，不保存用户照片。请只使用经本人明确同意的图片。")

register_tab, verify_tab, identify_tab, manage_tab, eval_tab = st.tabs(
    ["人员注册", "身份验证 1:1", "身份识别 1:N", "数据库与日志", "实验与说明"]
)

with register_tab:
    st.subheader("注册人脸特征")
    st.caption("为同一人注册 2～5 张不同角度的清晰照片，可提高实际使用时的容错能力。每次只保存一张人脸的向量。")
    col_a, col_b = st.columns([1, 1.2], gap="large")
    with col_a:
        name = st.text_input("人员姓名或本地代号", placeholder="例如：实验人员 A", key="reg_name", max_chars=50)
        reg_file = _photo_widget("register")
        consent = st.checkbox("我已取得照片中本人对采集和存储生物特征的明确同意", key="reg_consent")
        submit = st.button("提取特征并注册", type="primary", key="reg_btn")
    with col_b:
        if submit:
            try:
                image = _image_from_widget(reg_file)
                result, obs = service.register(name, image, consent=consent)
                st.success(f"已注册：{result['name']}；当前共有 {result['samples']} 条人脸样本")
                if obs.preview_jpeg:
                    st.image(obs.preview_jpeg, caption="检测框预览（不保存图片）", width=420)
                c1, c2 = st.columns(2)
                c1.metric("检测得分", f"{obs.detection_score:.3f}")
                c2.metric("特征提取耗时", f"{obs.encode_ms:.0f} ms")
            except (FaceInputError, PermissionError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
        else:
            st.info("录入人员代号并上传单人照片。重复使用同一代号可以追加人脸模板，最多 10 条。")

with verify_tab:
    st.subheader("1:1 身份验证")
    st.caption("选择一个已注册身份，比较当前照片与该人的全部已注册模板，不会搜索其他人的身份。")
    people = store.persons()
    if not people:
        st.warning("身份库为空。请先在“人员注册”中录入至少一个人。")
    else:
        options = {p["name"]: p for p in people}
        choice = st.selectbox("声称的身份", list(options), key="verify_person")
        verify_file = _photo_widget("verify")
        if st.button("开始验证", type="primary", key="verify_btn"):
            try:
                img = _image_from_widget(verify_file)
                result = service.verify(choice, img, threshold=threshold)
                if result.accepted:
                    st.success(f"验证通过：{result.name}")
                else:
                    st.error("验证失败：未达到指定相似度阈值")
                c1, c2, c3 = st.columns(3)
                c1.metric("余弦相似度", f"{result.similarity:.4f}" if result.similarity is not None else "无")
                c2.metric("判定阈值", f"{result.threshold:.2f}")
                c3.metric("总耗时", f"{result.elapsed_ms:.0f} ms")
                if result.observation.preview_jpeg:
                    st.image(result.observation.preview_jpeg, width=420, caption="上传图像中的单张人脸")
            except (FaceInputError, ValueError, RuntimeError) as exc:
                st.error(str(exc))

with identify_tab:
    st.subheader("1:N 已注册身份检索")
    st.caption("在本机自愿注册的有限身份库内查找最相近的模板。对未达到阈值的查询返回“未知身份”。")
    identify_file = _photo_widget("identify")
    max_results = st.slider("显示前 K 个候选", min_value=1, max_value=5, value=3, key="id_top_k")
    if st.button("搜索已注册身份", type="primary", key="identify_btn"):
        try:
            img = _image_from_widget(identify_file)
            result = service.identify(img, threshold=threshold, top_k=max_results)
            if result.accepted:
                st.success(f"最相近的已注册身份：{result.name}")
            else:
                st.warning("未知身份：无已注册模板达到阈值")
            c1, c2 = st.columns(2)
            c1.metric("最相似得分", f"{result.similarity:.4f}" if result.similarity is not None else "无")
            c2.metric("检索总耗时", f"{result.elapsed_ms:.0f} ms")
            if result.matches:
                st.dataframe(
                    pd.DataFrame([{"候选身份": name, "余弦相似度": round(sim,4), "是否达标": sim >= threshold} for name,sim in result.matches]),
                    hide_index=True, use_container_width=True,
                )
            if result.observation.preview_jpeg:
                st.image(result.observation.preview_jpeg, width=420, caption="输入图片单人检测结果")
        except (FaceInputError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

with manage_tab:
    st.subheader("人员与审计记录")
    st.caption("人员删除会同步删除其特征模板及以该名字记录的相关审计事件。为保护隐私，默认不在数据库中保存照片。")
    data = store.persons()
    if data:
        st.dataframe(
            pd.DataFrame([{"身份代号": p["name"], "模板数量": p["samples"], "注册时间(UTC)": p["created_utc"]} for p in data]),
            hide_index=True, use_container_width=True,
        )
        person_map = {p["name"]: p for p in data}
        target = st.selectbox("删除身份", ["请选择"] + list(person_map), key="delete_person")
        confirmed = st.checkbox("确认删除身份及其全部模板", key="delete_confirm")
        if st.button("删除所选身份及其模板", disabled=(target == "请选择" or not confirmed), key="delete_btn"):
            if store.delete_person(person_map[target]["person_id"]):
                st.success(f"已删除：{target}")
                st.rerun()
    else:
        st.info("尚无注册人员。")
    st.divider()
    logs = store.logs(limit=250)
    st.caption(f"最近 {len(logs)} 条认证与检索事件，不记录原始图片。")
    if logs:
        st.dataframe(pd.DataFrame(logs), hide_index=True, use_container_width=True)
    st.download_button("下载审计记录 CSV", store.export_logs_csv(), file_name="face_auth_events.csv", mime="text/csv")
    clear_confirm = st.checkbox("确认清空所有审计记录", key="clear_confirm")
    if st.button("清空记录", disabled=not clear_confirm, key="clear_btn"):
        number = store.clear_logs()
        st.success(f"已删除 {number} 条事件")
        st.rerun()

with eval_tab:
    from faceauth.dashboard import render_experiment_dashboard
    render_experiment_dashboard(Path(__file__).resolve().parent / "reports")
    st.divider()
    st.subheader("实验复现与安全边界")
    st.markdown("""
**工作流程**

1. YuNet 检测单张人脸，提取 5 个面部关键点；多张人脸直接拒绝。
2. SFace `alignCrop` 对齐人脸并提取深度特征向量，做 L2 归一化。
3. 注册端仅保存特征模板到 SQLite；验证使用 1:1 最大余弦相似度，检索执行已注册库的 Top-K 搜索。
4. 阈值必须通过独立验证集校准，报告 Accuracy、FAR、FRR 和 ROC-AUC。

**功能定位：** 本项目是本机的算法实验与课程/简历演示工程。上传普通照片或摄像头拍照均可被照片翻拍攻击。当前未实现可靠活体检测、双因素认证、硬件安全模块、数据库加密、管理员权限控制，因此不能作为真实门禁、支付、考勤或生产身份认证系统。

**复现命令**
""")
    st.code("""python scripts/download_models.py
python scripts/verify_env.py
python -m pytest -q
python scripts/smoke_test.py
python scripts/eval_pairs.py --help
streamlit run app.py --server.fileWatcherType none""", language="powershell")
    st.caption("模型来源：OpenCV Zoo，YuNet（MIT）与 SFace（Apache-2.0）。模型权重与测试照片不随源码发布。")
