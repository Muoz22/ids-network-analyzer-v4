# ================================================================
# app.py — AI Agents IDS v4
# Universal Network Intrusion Detection System
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, tempfile, json
from datetime import datetime

st.set_page_config(
    page_title="AI Agents IDS v4",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,
        #0a0a1a 0%, #0d1b2a 50%, #1a0a2e 100%);
    padding: 2rem; border-radius: 12px;
    margin-bottom: 2rem; text-align: center;
    border: 1px solid #00d4ff33;
}
.main-header h1 {
    color:#00d4ff; font-size:2.5rem; margin:0;
    text-shadow: 0 0 20px #00d4ff66;
}
.main-header p {
    color:#a0aec0; font-size:1rem;
    margin:0.5rem 0 0;
}
.metric-value {
    font-size:2rem; font-weight:bold; margin:0;
}
.metric-label {
    color:#a0aec0; font-size:0.85rem;
}
.status-excellent { color:#00d4ff; }
.status-good      { color:#48bb78; }
.status-warning   { color:#ed8936; }
.status-bad       { color:#e74c3c; }
.agent-card {
    border-left: 4px solid;
    padding: 1rem; margin: 0.5rem 0;
    background: rgba(0,0,0,0.05);
    border-radius: 0 8px 8px 0;
}
.dataset-badge {
    background: rgba(0,212,255,0.1);
    border: 1px solid #00d4ff44;
    border-radius: 8px; padding: 0.5rem 1rem;
    color: #00d4ff; font-weight: bold;
    display: inline-block; margin: 0.2rem;
}
.log-type-badge {
    background: rgba(155,89,182,0.15);
    border: 1px solid #9b59b666;
    border-radius: 8px; padding: 0.4rem 0.9rem;
    color: #9b59b6; font-weight: bold;
    display: inline-block; margin: 0.2rem;
}
.risk-high   { color:#e74c3c; font-weight:bold; }
.risk-medium { color:#f39c12; font-weight:bold; }
.risk-low    { color:#2ecc71; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Agents IDS v4</h1>
    <p>Universal Network Intrusion Detection —
    FT-Transformer | 5 Datasets |
    18 Abstract Features | Log Analysis</p>
    <p style="color:#666;font-size:0.8rem;
    margin-top:0.5rem;">
    © 2025 Muaz Al-Soufi |
    <a href="https://github.com/Muoz22/ids-network-analyzer-v4"
    style="color:#00d4ff;">GitHub v4</a>
    &nbsp;|&nbsp;
    <a href="https://github.com/Muoz22/ids-network-analyzer"
    style="color:#00d4ff;">GitHub v3</a>
    </p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# Load Models
# ================================================================

@st.cache_resource
def load_models_cached():
    try:
        from inference_v4 import load_models_v4
        models = load_models_v4(
            model_dir="models/",
            v4_dir="v4/")
        return models, None
    except Exception as e:
        return None, str(e)

models_global, models_err = load_models_cached()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ إعدادات التحليل")

    threshold = st.slider(
        "Unknown Threshold",
        min_value=0.3, max_value=0.9,
        value=0.6, step=0.05,
        help="احتمالية أقل من هذا = Unknown")

    st.markdown("---")
    st.markdown("### 🤖 النموذج")

    if models_global:
        meta = models_global.get("meta", {})
        st.success("✅ FT-Transformer v4")
        st.write(f"**Features:** "
                 f"{meta.get('n_features',18)} "
                 f"Abstract Behavioral")
        st.write(f"**Classes:** "
                 f"{meta.get('n_classes',10)}")
        st.write(f"**Accuracy:** "
                 f"{meta.get('accuracy',0)*100:.1f}%")

        st.markdown("**Trained on:**")
        for ds in meta.get("trained_on", []):
            st.markdown(
                f'<span class="dataset-badge">'
                f'{ds}</span>',
                unsafe_allow_html=True)

        with st.expander("🔍 Abstract Features"):
            for f in meta.get("feat_cols", []):
                st.write(f"  • {f}")
    elif models_err:
        st.error(f"❌ {models_err}")

    st.markdown("---")
    st.markdown("### 🔗 روابط")
    st.markdown(
        "[🐙 GitHub v4](https://github.com/"
        "Muoz22/ids-network-analyzer-v4)  \n"
        "[🐙 GitHub v3](https://github.com/"
        "Muoz22/ids-network-analyzer)  \n"
        "[🎓 Google Scholar](https://scholar"
        ".google.com/citations?"
        "user=J35vcAIAAAAJ&hl=en)")


# ── 6 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 تحليل الشبكة",
    "📊 تقرير تفصيلي",
    "🔬 قابلية التفسير",
    "📁 تحليل Log Files",
    "📖 كيف يعمل",
    "📋 عن المشروع",
])


# ══════════════════════════════════════════════════════════════
# Tab 1 — تحليل الشبكة
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📁 ارفع ملف CSV للتحليل")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "اختر ملف CSV — أي داتاست شبكية",
            type=["csv"],
            key="uploader_tab1")
    with col2:
        st.markdown("""
        **الداتاسيت المدعومة تلقائياً:**
        - ✅ TON-IoT
        - ✅ CIC-IoT-2023
        - ✅ UNSW-NB15
        - ✅ Bot-IoT
        - ✅ CICIDS2017
        - ✅ أي CSV بـ network features
        """)

    if uploaded_file is not None:
        with st.spinner("جاري قراءة الملف..."):
            try:
                df = pd.read_csv(
                    uploaded_file,
                    low_memory=False)
                st.success(
                    f"✅ {df.shape[0]:,} صف × "
                    f"{df.shape[1]} عمود")
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

        with st.expander(
                "👁️ معاينة البيانات",
                expanded=False):
            st.dataframe(
                df.head(10),
                use_container_width=True)

        if st.button(
                "🚀 ابدأ التحليل",
                type="primary",
                use_container_width=True,
                key="btn_analyze_v4"):

            for k in ["results_v4",
                      "plot_bytes_v4",
                      "exp_bytes_v4"]:
                if k in st.session_state:
                    del st.session_state[k]

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🔮 جاري التحليل..."):

                from inference_v4 import (
                    run_inference_v4,
                    make_plots_v4,
                    make_explainability_v4)

                if not models_global:
                    st.error("❌ النماذج غير محملة")
                    st.stop()

                status.text(
                    "🤖 تشغيل FT-Transformer v4...")
                progress.progress(20)

                results = run_inference_v4(
                    df, models_global,
                    ft_unk_thr=threshold)

                status.text("📊 إنتاج الرسوم...")
                progress.progress(50)

                with tempfile.TemporaryDirectory() \
                        as tmp:
                    plot_paths = make_plots_v4(
                        results,
                        out_dir=tmp,
                        models_ref=models_global)
                    plot_bytes = []
                    for title, path in plot_paths:
                        try:
                            with open(
                                    path, "rb") as f:
                                plot_bytes.append(
                                    (title,
                                     f.read()))
                        except Exception:
                            pass

                status.text(
                    "🔬 حساب قابلية التفسير...")
                progress.progress(80)

                exp_bytes = []
                with tempfile.TemporaryDirectory() \
                        as tmp2:
                    try:
                        exp_paths = \
                            make_explainability_v4(
                                results,
                                models_global,
                                out_dir=tmp2)
                        for title, path in \
                                exp_paths:
                            try:
                                with open(
                                        path,
                                        "rb") as f:
                                    exp_bytes\
                                        .append(
                                        (title,
                                         f.read()))
                            except Exception:
                                pass
                    except Exception:
                        pass

                progress.progress(100)
                status.text("✅ اكتمل!")

                st.session_state[
                    "results_v4"]    = results
                st.session_state[
                    "plot_bytes_v4"] = plot_bytes
                st.session_state[
                    "exp_bytes_v4"]  = exp_bytes

                st.markdown("---")
                st.markdown("## 🏆 نتائج التحليل")

                ds = results.get(
                    "detected_ds", "AUTO")
                st.markdown(
                    f'**Dataset detected:** '
                    f'<span class="dataset-badge">'
                    f'{ds}</span>',
                    unsafe_allow_html=True)

                total   = results["n_samples"]
                benign  = results["n_benign"]
                attacks = results["n_attacks"]
                unknown = results["n_unknown"]
                m       = results["metrics"]

                st.markdown(
                    "### 🎯 Level 1 — "
                    "Benign / Attack")
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("إجمالي",
                              f"{total:,}")
                with c2:
                    st.metric(
                        "✅ Benign",
                        f"{benign:,}",
                        f"{100*benign/total:.1f}%")
                with c3:
                    st.metric(
                        "🚫 Attack",
                        f"{attacks:,}",
                        f"{100*attacks/total:.1f}%",
                        delta_color="inverse")
                with c4:
                    st.metric(
                        "⚠️ Unknown",
                        f"{unknown:,}",
                        f"{100*unknown/total:.1f}%",
                        delta_color="inverse")

                if "l1_accuracy" in m:
                    st.markdown(
                        f"**Level 1 Accuracy:** "
                        f"`{m['l1_accuracy']*100:.2f}%`"
                        f"  |  **F1:** "
                        f"`{m['l1_f1']*100:.2f}%`")

                st.markdown(
                    "### 🔍 Level 2 — "
                    "Attack Family")
                if "accuracy" in m:
                    mc1,mc2,mc3 = st.columns(3)
                    for col,(name,val,hi,lo) \
                            in zip(
                            [mc1,mc2,mc3],[
                            ("Accuracy",
                             m["accuracy"]*100,
                             93,85),
                            ("Weighted F1",
                             m["weighted_f1"]*100,
                             90,80),
                            ("Macro F1",
                             m["macro_f1"]*100,
                             80,60)]):
                        with col:
                            cl = (
                                "status-excellent"
                                if val>hi else
                                "status-good"
                                if val>lo else
                                "status-warning")
                            st.markdown(
                                f'<p class="metric-value'
                                f' {cl}">'
                                f'{val:.2f}%</p>'
                                f'<p class="metric-label">'
                                f'{name}</p>',
                                unsafe_allow_html=True)

                if results["atk_counts"]:
                    st.markdown(
                        "### 🔴 أنواع الهجمات")
                    atk_df = pd.DataFrame(
                        results[
                            "atk_counts"
                        ].most_common(15),
                        columns=[
                            "نوع الهجوم",
                            "العدد"])
                    atk_df["النسبة %"] = (
                        atk_df["العدد"] /
                        total * 100).round(2)
                    st.dataframe(
                        atk_df,
                        use_container_width=True)

                if plot_bytes:
                    st.markdown(
                        "### 📊 الرسوم البيانية")
                    for i in range(
                            0, len(plot_bytes), 2):
                        cols = st.columns(2)
                        for j, (title, img) \
                                in enumerate(
                                plot_bytes[i:i+2]):
                            with cols[j]:
                                st.markdown(
                                    f"**{title}**")
                                st.image(
                                    img,
                                    use_column_width=True)

                if "report" in m:
                    with st.expander(
                            "📋 Classification "
                            "Report"):
                        st.code(m["report"])

                st.markdown("### 💾 تحميل النتائج")
                result_df = pd.DataFrame({
                    "prediction_l2":
                        results["y_pred_l2"],
                    "prediction_l1":
                        results["y_pred_l1"],
                    "confidence":
                        results["y_conf"],
                    "is_unknown":
                        results["y_unknown"],
                })
                st.download_button(
                    "⬇️ تحميل النتائج CSV",
                    result_df.to_csv(index=False),
                    f"ids_v4_results_"
                    f"{datetime.now():%Y%m%d_%H%M}"
                    f".csv",
                    "text/csv",
                    use_container_width=True,
                    key="dl_tab1")

                st.info(
                    "💡 انتقل لـ "
                    "**📊 تقرير تفصيلي** و"
                    "**🔬 قابلية التفسير** "
                    "لرؤية كل التفاصيل")


# ══════════════════════════════════════════════════════════════
# Tab 2 — تقرير تفصيلي
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 تقرير تفصيلي")

    if "results_v4" not in st.session_state:
        st.info(
            "🔍 ارفع ملف في "
            "**تحليل الشبكة** أولاً")
    else:
        results    = st.session_state["results_v4"]
        plot_bytes = st.session_state["plot_bytes_v4"]
        m          = results["metrics"]
        total      = results["n_samples"]
        benign     = results["n_benign"]
        attacks    = results["n_attacks"]
        unknown    = results["n_unknown"]
        ds         = results.get("detected_ds","AUTO")

        st.markdown(
            f'**Dataset:** '
            f'<span class="dataset-badge">'
            f'{ds}</span>  '
            f'**Samples:** {total:,}',
            unsafe_allow_html=True)

        st.markdown(
            "### 🎯 Level 1 — Benign/Attack")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric("✅ ALLOW", f"{benign:,}",
                      f"{100*benign/total:.1f}%")
        with c2:
            st.metric("🚫 BLOCK", f"{attacks:,}",
                      f"{100*attacks/total:.1f}%",
                      delta_color="inverse")
        with c3:
            st.metric("⚠️ QUARANTINE",
                      f"{unknown:,}",
                      f"{100*unknown/total:.1f}%",
                      delta_color="inverse")

        if "l1_accuracy" in m:
            st.success(
                f"Level 1 — "
                f"Accuracy: "
                f"{m['l1_accuracy']*100:.2f}%  |  "
                f"F1: {m['l1_f1']*100:.2f}%")

        st.markdown("---")
        st.markdown("### 🎯 Decision Analysis")
        pie_imgs = [(t,img) for t,img in plot_bytes
                    if "Distribution" in t]
        atk_imgs = [(t,img) for t,img in plot_bytes
                    if "Attack" in t]
        dc1, dc2 = st.columns(2)
        with dc1:
            if pie_imgs:
                st.markdown(
                    "**Decision Distribution**")
                st.image(pie_imgs[0][1],
                         use_column_width=True)
        with dc2:
            if atk_imgs:
                st.markdown("**Attack Types**")
                st.image(atk_imgs[0][1],
                         use_column_width=True)

        if "accuracy" in m:
            st.markdown("---")
            st.markdown(
                "### 📈 Level 2 Performance")
            mc1,mc2,mc3 = st.columns(3)
            for col,(name,val,hi,lo) in zip(
                    [mc1,mc2,mc3],[
                    ("Accuracy",
                     m["accuracy"]*100,93,85),
                    ("Weighted F1",
                     m["weighted_f1"]*100,90,80),
                    ("Macro F1",
                     m["macro_f1"]*100,80,60)]):
                with col:
                    cl = ("status-excellent"
                          if val>hi else
                          "status-good"
                          if val>lo else
                          "status-warning")
                    st.markdown(
                        f'<p class="metric-value'
                        f' {cl}">{val:.2f}%</p>'
                        f'<p class="metric-label">'
                        f'{name}</p>',
                        unsafe_allow_html=True)

            conf_imgs  = [(t,img)
                          for t,img in plot_bytes
                          if "Confusion" in t]
            train_imgs = [(t,img)
                          for t,img in plot_bytes
                          if "Training" in t]
            if conf_imgs or train_imgs:
                pf1, pf2 = st.columns(2)
                with pf1:
                    if conf_imgs:
                        st.markdown(
                            "**Confusion Matrix**")
                        st.image(conf_imgs[0][1],
                                 use_column_width=True)
                with pf2:
                    if train_imgs:
                        st.markdown(
                            "**Training Curves**")
                        st.image(train_imgs[0][1],
                                 use_column_width=True)

            f1_imgs = [(t,img)
                       for t,img in plot_bytes
                       if "F1" in t]
            if f1_imgs:
                st.markdown("**Per-Class F1**")
                st.image(f1_imgs[0][1],
                         use_column_width=True)

            if "report" in m:
                st.markdown("---")
                st.markdown(
                    "### 📋 Classification Report")
                st.code(m["report"])

        st.markdown("---")
        st.markdown("### 🎚️ Confidence Analysis")
        conf_imgs = [(t,img) for t,img in plot_bytes
                     if t == "Confidence"]
        if conf_imgs:
            st.image(conf_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        st.markdown("### 🌊 Drift Analysis")
        drift_imgs = [(t,img) for t,img in plot_bytes
                      if "Drift" in t]
        if drift_imgs:
            st.image(drift_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        dash_imgs = [(t,img) for t,img in plot_bytes
                     if "Dashboard" in t]
        if dash_imgs:
            st.markdown("### 🗂️ Summary Dashboard")
            st.image(dash_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        st.markdown("### 🖼️ كل الرسوم البيانية")
        for i in range(0, len(plot_bytes), 2):
            cols = st.columns(2)
            for j, (title, img) in enumerate(
                    plot_bytes[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(img,
                             use_column_width=True)

        st.markdown("---")
        result_df = pd.DataFrame({
            "prediction_l2": results["y_pred_l2"],
            "prediction_l1": results["y_pred_l1"],
            "confidence"   : results["y_conf"],
            "is_unknown"   : results["y_unknown"],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ تحميل النتائج CSV",
                result_df.to_csv(index=False),
                f"ids_v4_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_tab2")
        with c2:
            if "report" in m:
                st.download_button(
                    "⬇️ تحميل التقرير TXT",
                    m["report"],
                    f"report_v4_"
                    f"{datetime.now():%Y%m%d_%H%M}.txt",
                    "text/plain",
                    use_container_width=True,
                    key="dl_tab2_txt")


# ══════════════════════════════════════════════════════════════
# Tab 3 — قابلية التفسير
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔬 قابلية التفسير")

    if "exp_bytes_v4" not in st.session_state:
        st.info(
            "🔍 ارفع ملف في "
            "**تحليل الشبكة** أولاً")
    else:
        exp_bytes = st.session_state["exp_bytes_v4"]
        results   = st.session_state["results_v4"]
        ds        = results.get("detected_ds","AUTO")

        st.markdown(
            f'**Dataset:** '
            f'<span class="dataset-badge">'
            f'{ds}</span>  '
            f'**Samples:** '
            f'{results["n_samples"]:,}',
            unsafe_allow_html=True)

        st.markdown("""
        > تُحسب قيم SHAP من البيانات الفعلية
        للداتاست المرفوعة — تتغير مع كل ملف جديد.
        """)

        if not exp_bytes:
            st.warning("⚠️ لم تُنتج رسوم التفسير")
        else:
            train_exp = [(t,img)
                         for t,img in exp_bytes
                         if "Training" in t]
            if train_exp:
                st.markdown(
                    "### 📈 Training Curves "
                    "— FT-Transformer v4")
                st.markdown(
                    "*النموذج تدرّب على 5 "
                    "داتاسيت مختلفة*")
                st.image(train_exp[0][1],
                         use_column_width=True)

            st.markdown("---")

            shap_s = [(t,img) for t,img in exp_bytes
                      if "SHAP Summary" in t]
            shap_w = [(t,img) for t,img in exp_bytes
                      if "Waterfall" in t]

            if shap_s or shap_w:
                st.markdown("### 🧠 SHAP Analysis")
                st.markdown(
                    "*تأثير كل feature على "
                    "قرار النموذج — "
                    "محسوب من بياناتك الفعلية*")
                sc1, sc2 = st.columns(2)
                with sc1:
                    if shap_s:
                        st.markdown(
                            "**SHAP Summary "
                            "— Feature Impact**")
                        st.image(shap_s[0][1],
                                 use_column_width=True)
                with sc2:
                    if shap_w:
                        st.markdown(
                            "**SHAP Waterfall "
                            "— Cumulative**")
                        st.image(shap_w[0][1],
                                 use_column_width=True)

            st.markdown("---")
            other_exp = [
                (t,img) for t,img in exp_bytes
                if "Training" not in t
                and "SHAP" not in t]
            if other_exp:
                st.markdown("### 📊 تحليل إضافي")
                for i in range(
                        0, len(other_exp), 2):
                    cols = st.columns(2)
                    for j, (title, img) in \
                            enumerate(
                            other_exp[i:i+2]):
                        with cols[j]:
                            st.markdown(
                                f"**{title}**")
                            st.image(
                                img,
                                use_column_width=True)

        st.markdown("---")
        st.markdown(
            "### 📋 Feature Importance Table")
        feat_names = results.get("feat_names", [])
        X          = results["X_final"]
        if len(feat_names) > 0:
            imp = np.abs(X).mean(axis=0)
            std = np.abs(X).std(axis=0)
            idx = np.argsort(imp)[::-1]
            feat_df = pd.DataFrame({
                "Feature": [feat_names[i]
                             for i in idx],
                "Importance": [f"{imp[i]:.4f}"
                               for i in idx],
                "Std": [f"{std[i]:.4f}"
                        for i in idx],
            })
            st.dataframe(feat_df,
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Tab 4 — تحليل Log Files
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📁 تحليل Log Files")
    st.markdown("""
    ارفع أي ملف log — يُكتشف النوع تلقائياً
    ويُحلَّل بالنموذج.
    """)

    # ── اختيار طريقة الإدخال ──────────────────────────────────
    input_method = st.radio(
        "طريقة الإدخال:",
        ["📂 ارفع ملف (.log/.txt/.csv)",
         "📋 الصق نص مباشرة"],
        horizontal=True,
        key="log_input_method")

    log_content = None

    if input_method == \
            "📂 ارفع ملف (.log/.txt/.csv)":
        log_file = st.file_uploader(
            "ارفع ملف Log",
            type=["log","txt","csv","evtx"],
            key="log_uploader")
        if log_file:
            try:
                raw = log_file.read()
                try:
                    log_content = raw.decode(
                        "utf-8")
                except UnicodeDecodeError:
                    log_content = raw.decode(
                        "latin-1")
                st.success(
                    f"✅ تم تحميل: "
                    f"{log_file.name} — "
                    f"{len(log_content.splitlines()):,}"
                    f" سطر")
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        log_content = st.text_area(
            "الصق محتوى الـ Log هنا:",
            height=200,
            placeholder=(
                "Jan 15 10:23:45 server sshd[1234]:"
                " Failed password for root from "
                "192.168.1.100 port 22 ssh2\n"
                "Jan 15 10:23:46 server sshd[1234]:"
                " Failed password for admin from "
                "192.168.1.100 port 22 ssh2\n"
                "..."),
            key="log_text_area")

    # ── معلومات أنواع الـ Logs ─────────────────────────────────
    with st.expander(
            "📖 أنواع الـ Logs المدعومة",
            expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🔐 SSH Logs**
            - `/var/log/auth.log`
            - `/var/log/secure`
            - OpenSSH logs

            **🌐 Web Server**
            - Apache access/error log
            - Nginx access log
            - Combined log format
            """)
        with col2:
            st.markdown("""
            **🖥️ Windows Events**
            - Security log (CSV export)
            - Event ID 4624/4625
            - Logon/Logoff events

            **🔥 Firewall**
            - iptables logs
            - Cisco ASA
            - Palo Alto logs
            """)
        with col3:
            st.markdown("""
            **🌍 DNS Logs**
            - BIND named logs
            - DNS query logs
            - Response codes

            **📋 Generic**
            - Syslog format
            - Any text log
            - Custom formats
            """)

    # ── زر التحليل ────────────────────────────────────────────
    if log_content and len(
            log_content.strip()) > 0:

        n_lines = len(
            log_content.splitlines())
        st.info(
            f"📋 {n_lines:,} سطر جاهز للتحليل")

        max_lines = st.slider(
            "الحد الأقصى للسطور:",
            min_value=100,
            max_value=50000,
            value=min(n_lines, 5000),
            step=100,
            key="log_max_lines")

        if st.button(
                "🔍 ابدأ تحليل الـ Log",
                type="primary",
                use_container_width=True,
                key="btn_analyze_log"):

            for k in ["log_results",
                      "log_parser_obj"]:
                if k in st.session_state:
                    del st.session_state[k]

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner(
                    "🔮 جاري تحليل الـ Log..."):

                try:
                    import sys, importlib.util
                    sys.path.insert(0, ".")
                    spec = importlib.util\
                        .spec_from_file_location(
                        "log_parser_v4",
                        "v4/log_parser_v4.py")
                    _lp = importlib.util\
                        .module_from_spec(spec)
                    spec.loader.exec_module(_lp)
                    UniversalLogParser = \
                        _lp.UniversalLogParser

                    # ── Step 1: Parse ──────────
                    status.text(
                        "📋 تحليل الـ Log...")
                    progress.progress(20)

                    parser = UniversalLogParser()
                    df_feat = parser.parse_file(
                        log_content,
                        max_lines=max_lines)

                    if df_feat.empty:
                        st.error(
                            "❌ لم يُتمكن من "
                            "تحليل الـ Log")
                        st.stop()

                    # ── Step 2: ML Inference ───
                    status.text(
                        "🤖 تشغيل النموذج...")
                    progress.progress(50)

                    log_results = {
                        "log_type"   : parser.log_type,
                        "stats"      : parser.stats,
                        "n_lines"    : len(
                            parser.parsed_lines),
                        "predictions": [],
                        "atk_counts" : {},
                        "risk_counts": {
                            "HIGH":0,
                            "MEDIUM":0,
                            "LOW":0},
                    }

                    if models_global and \
                            len(df_feat) > 0:
                        from inference_v4 import \
                            run_inference_v4

                        # نعمل inference على
                        # الـ features المحسوبة
                        X_sc = models_global[
                            "scaler"].transform(
                            df_feat.values.astype(
                                np.float32))

                        batch_size = 4096
                        all_probs  = []
                        for i in range(
                                0, len(X_sc),
                                batch_size):
                            batch = X_sc[
                                i:i+batch_size
                            ].astype(np.float32)
                            probs = models_global[
                                "session"].run(
                                [models_global[
                                    "output_name"]],
                                {models_global[
                                    "input_name"]:
                                    batch})[0]
                            all_probs.append(probs)

                        y_probs = np.vstack(
                            all_probs)
                        y_pred  = np.argmax(
                            y_probs, axis=1)
                        y_conf  = y_probs.max(
                            axis=1)
                        y_unk   = y_conf < threshold

                        class_names = \
                            models_global[
                                "class_names"]
                        level1_map  = \
                            models_global[
                                "level1_map"]

                        y_pred_l2 = [
                            class_names[i]
                            if i < len(class_names)
                            else "unknown"
                            for i in y_pred]
                        y_pred_l1 = [
                            level1_map.get(
                                p, "attack")
                            for p in y_pred_l2]

                        log_results[
                            "y_pred_l2"] = y_pred_l2
                        log_results[
                            "y_pred_l1"] = y_pred_l1
                        log_results[
                            "y_conf"]    = y_conf
                        log_results[
                            "y_unk"]     = y_unk

                        from collections import Counter
                        log_results["atk_counts"] = \
                            Counter(
                                p for p, u in zip(
                                    y_pred_l2, y_unk)
                                if p != "benign"
                                and not u)

                        total_l = len(y_pred_l2)
                        benign_l = sum(
                            1 for p in y_pred_l2
                            if p == "benign")
                        attack_l = total_l - benign_l
                        log_results[
                            "n_total"]  = total_l
                        log_results[
                            "n_benign"] = benign_l
                        log_results[
                            "n_attack"] = attack_l

                    # ── Step 3: Suspicious IPs ─
                    status.text(
                        "🔍 تحليل الـ IPs...")
                    progress.progress(75)

                    susp_ips = \
                        parser.get_suspicious_ips()
                    timeline = parser.get_timeline()

                    log_results[
                        "suspicious_ips"] = susp_ips
                    log_results[
                        "timeline"] = timeline

                    # ── Step 4: Plots ──────────
                    status.text(
                        "📊 إنتاج الرسوم...")
                    progress.progress(90)

                    import matplotlib.pyplot as plt
                    import matplotlib.gridspec \
                        as gridspec
                    log_plots = []

                    with tempfile.TemporaryDirectory(
                    ) as tmp_log:

                        # Plot 1: Attack Types
                        try:
                            if log_results.get(
                                    "atk_counts"):
                                fig, ax = plt.subplots(
                                    figsize=(10, 5))
                                top = sorted(
                                    log_results[
                                        "atk_counts"
                                    ].items(),
                                    key=lambda x:
                                    -x[1])[:10]
                                nms = [x[0]
                                       for x in top]
                                vls = [x[1]
                                       for x in top]
                                colors = [
                                    "#e74c3c"
                                    if v > max(vls)*0.5
                                    else "#f39c12"
                                    for v in vls]
                                ax.barh(
                                    nms, vls,
                                    color=colors,
                                    alpha=0.85)
                                for i, v in \
                                        enumerate(vls):
                                    ax.text(
                                        v+max(vls)*0.01,
                                        i, f"{v:,}",
                                        va="center",
                                        fontsize=9)
                                ax.set_xlabel("Count")
                                ax.set_title(
                                    f"Attack Types "
                                    f"Detected — "
                                    f"{parser.log_type}"
                                    f" Log",
                                    fontweight="bold")
                                ax.grid(
                                    axis="x",
                                    alpha=0.3)
                                plt.tight_layout()
                                p = os.path.join(
                                    tmp_log,
                                    "log_attacks.png")
                                fig.savefig(
                                    p,
                                    bbox_inches="tight")
                                plt.close()
                                with open(p,"rb") as f:
                                    log_plots.append(
                                        ("Attack Types",
                                         f.read()))
                        except Exception:
                            pass

                        # Plot 2: Action Distribution
                        try:
                            actions = parser.stats.get(
                                "actions", {})
                            if actions:
                                fig, ax = plt.subplots(
                                    figsize=(7, 6))
                                colors_a = {
                                    "FAIL"   :"#e74c3c",
                                    "DENY"   :"#c0392b",
                                    "SUCCESS":"#2ecc71",
                                    "ALLOW"  :"#27ae60",
                                    "INFO"   :"#3498db",
                                }
                                ax.pie(
                                    list(
                                        actions.values()),
                                    labels=list(
                                        actions.keys()),
                                    colors=[
                                        colors_a.get(
                                            k,"#95a5a6")
                                        for k in
                                        actions.keys()],
                                    autopct="%1.1f%%",
                                    startangle=140,
                                    shadow=True)
                                ax.set_title(
                                    "Action "
                                    "Distribution",
                                    fontweight="bold")
                                plt.tight_layout()
                                p = os.path.join(
                                    tmp_log,
                                    "log_actions.png")
                                fig.savefig(
                                    p,
                                    bbox_inches="tight")
                                plt.close()
                                with open(p,"rb") as f:
                                    log_plots.append(
                                        ("Actions",
                                         f.read()))
                        except Exception:
                            pass

                        # Plot 3: Top IPs heatmap
                        try:
                            if not susp_ips.empty \
                                    and len(
                                    susp_ips) > 0:
                                fig, ax = plt.subplots(
                                    figsize=(10, 6))
                                top_ips = susp_ips.head(
                                    10)
                                risk_colors = {
                                    "🔴 High"  :"#e74c3c",
                                    "🟡 Medium":"#f39c12",
                                    "🟢 Low"   :"#2ecc71",
                                }
                                bar_colors = [
                                    risk_colors.get(
                                        r,"#95a5a6")
                                    for r in
                                    top_ips[
                                        "Risk Level"]]
                                ax.barh(
                                    top_ips["IP"],
                                    top_ips[
                                        "Requests"
                                    ].astype(int),
                                    color=bar_colors,
                                    alpha=0.85)
                                ax.set_xlabel(
                                    "Requests")
                                ax.set_title(
                                    "Top Suspicious IPs",
                                    fontweight="bold")
                                ax.grid(
                                    axis="x",
                                    alpha=0.3)
                                plt.tight_layout()
                                p = os.path.join(
                                    tmp_log,
                                    "log_ips.png")
                                fig.savefig(
                                    p,
                                    bbox_inches="tight")
                                plt.close()
                                with open(p,"rb") as f:
                                    log_plots.append(
                                        ("Suspicious IPs",
                                         f.read()))
                        except Exception:
                            pass

                        # Plot 4: Attack Patterns
                        try:
                            patterns = parser.stats.get(
                                "attack_patterns", {})
                            if patterns:
                                fig, ax = plt.subplots(
                                    figsize=(10, 5))
                                pats = sorted(
                                    patterns.items(),
                                    key=lambda x:
                                    -x[1])
                                pnames = [p[0]
                                          for p in pats]
                                pcounts= [p[1]
                                          for p in pats]
                                ax.bar(
                                    range(len(pnames)),
                                    pcounts,
                                    color="#e74c3c",
                                    alpha=0.85)
                                ax.set_xticks(
                                    range(len(pnames)))
                                ax.set_xticklabels(
                                    pnames,
                                    rotation=30,
                                    ha="right")
                                ax.set_ylabel("Count")
                                ax.set_title(
                                    "Attack Patterns "
                                    "Detected",
                                    fontweight="bold")
                                ax.grid(
                                    axis="y",
                                    alpha=0.3)
                                plt.tight_layout()
                                p = os.path.join(
                                    tmp_log,
                                    "log_patterns.png")
                                fig.savefig(
                                    p,
                                    bbox_inches="tight")
                                plt.close()
                                with open(p,"rb") as f:
                                    log_plots.append(
                                        ("Attack "
                                         "Patterns",
                                         f.read()))
                        except Exception:
                            pass

                        # Plot 5: Severity Timeline
                        try:
                            if not timeline.empty \
                                    and len(
                                    timeline) > 0:
                                fig, ax = plt.subplots(
                                    figsize=(14, 4))
                                sev_colors = {
                                    "HIGH"  :"#e74c3c",
                                    "MEDIUM":"#f39c12",
                                    "LOW"   :"#2ecc71",
                                }
                                for sev, grp in \
                                        timeline\
                                        .groupby(
                                        "severity"):
                                    ax.scatter(
                                        grp.index,
                                        [sev]*len(grp),
                                        c=sev_colors.get(
                                            sev,
                                            "#95a5a6"),
                                        alpha=0.6, s=20,
                                        label=sev)
                                ax.set_xlabel(
                                    "Event Index")
                                ax.set_ylabel(
                                    "Severity")
                                ax.set_title(
                                    "Event Severity "
                                    "Timeline",
                                    fontweight="bold")
                                ax.legend()
                                ax.grid(alpha=0.3)
                                plt.tight_layout()
                                p = os.path.join(
                                    tmp_log,
                                    "log_timeline.png")
                                fig.savefig(
                                    p,
                                    bbox_inches="tight")
                                plt.close()
                                with open(p,"rb") as f:
                                    log_plots.append(
                                        ("Severity "
                                         "Timeline",
                                         f.read()))
                        except Exception:
                            pass

                    log_results["plots"] = log_plots
                    st.session_state[
                        "log_results"] = log_results
                    st.session_state[
                        "log_parser_obj"] = parser

                    progress.progress(100)
                    status.text("✅ اكتمل!")

                except Exception as e:
                    st.error(
                        f"❌ خطأ في التحليل: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ── عرض النتائج ───────────────────────────────────────
        if "log_results" in st.session_state:
            lr = st.session_state["log_results"]

            st.markdown("---")
            st.markdown("## 🏆 نتائج تحليل الـ Log")

            # Log Type Badge
            lt = lr.get("log_type","GENERIC")
            st.markdown(
                f'**Log Type Detected:** '
                f'<span class="log-type-badge">'
                f'{lt}</span>',
                unsafe_allow_html=True)

            # إحصاءات أساسية
            stats = lr.get("stats", {})
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric(
                    "📋 إجمالي السطور",
                    f"{stats.get('total_lines',0):,}")
            with c2:
                st.metric(
                    "🌐 IPs فريدة",
                    f"{stats.get('unique_ips',0):,}")
            with c3:
                st.metric(
                    "❌ أحداث فاشلة",
                    f"{stats.get('fail_count',0):,}")
            with c4:
                n_patterns = len(
                    stats.get(
                        "attack_patterns", {}))
                st.metric(
                    "⚠️ أنماط هجوم",
                    f"{n_patterns}")

            # ML Results
            if "n_total" in lr:
                st.markdown("---")
                st.markdown(
                    "### 🤖 تصنيف النموذج")
                mc1,mc2,mc3 = st.columns(3)
                n_tot = max(lr["n_total"], 1)
                with mc1:
                    pct = 100*lr["n_benign"]/n_tot
                    st.metric(
                        "✅ Benign",
                        f"{lr['n_benign']:,}",
                        f"{pct:.1f}%")
                with mc2:
                    pct = 100*lr["n_attack"]/n_tot
                    st.metric(
                        "🚫 Attack",
                        f"{lr['n_attack']:,}",
                        f"{pct:.1f}%",
                        delta_color="inverse")
                with mc3:
                    atk_types = len(
                        lr.get("atk_counts", {}))
                    st.metric(
                        "🔍 أنواع هجمات",
                        f"{atk_types}")
                if lr.get("atk_counts"):
                    st.markdown(
                        "**أنواع الهجمات المكتشفة:**")
                    atk_df = pd.DataFrame(
                        sorted(
                            lr["atk_counts"].items(),
                            key=lambda x: -x[1]),
                        columns=["النوع","العدد"])
                    st.dataframe(
                        atk_df,
                        use_container_width=True)

            # Plots
            plots = lr.get("plots", [])
            if plots:
                st.markdown("---")
                st.markdown(
                    "### 📊 الرسوم البيانية")
                for i in range(
                        0, len(plots), 2):
                    cols = st.columns(2)
                    for j, (title, img) in \
                            enumerate(plots[i:i+2]):
                        with cols[j]:
                            st.markdown(
                                f"**{title}**")
                            st.image(
                                img,
                                use_column_width=True)

            # Suspicious IPs Table
            susp = lr.get("suspicious_ips",
                          pd.DataFrame())
            if not susp.empty:
                st.markdown("---")
                st.markdown(
                    "### 🔴 الـ IPs المشبوهة")
                st.dataframe(
                    susp,
                    use_container_width=True)

            # Attack Patterns
            patterns = stats.get(
                "attack_patterns", {})
            if patterns:
                st.markdown("---")
                st.markdown(
                    "### ⚠️ أنماط الهجوم المكتشفة")
                for pat, count in sorted(
                        patterns.items(),
                        key=lambda x: -x[1]):
                    severity = (
                        "🔴" if count > 10
                        else "🟡" if count > 3
                        else "🟢")
                    st.markdown(
                        f"{severity} **{pat}** "
                        f"— {count:,} مرة")

            # Download Report
            st.markdown("---")
            st.markdown("### 💾 تحميل التقرير")

            report_lines = [
                f"AI Agents IDS v4 — Log Analysis Report",
                f"Generated: "
                f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                f"="*50,
                f"Log Type: {lt}",
                f"Total Lines: "
                f"{stats.get('total_lines',0):,}",
                f"Unique IPs: "
                f"{stats.get('unique_ips',0):,}",
                f"Failed Events: "
                f"{stats.get('fail_count',0):,}",
                f"Attack Patterns: "
                f"{len(patterns)}",
                f"="*50,
            ]

            if "n_total" in lr:
                report_lines += [
                    f"ML Classification:",
                    f"  Benign: "
                    f"{lr['n_benign']:,}",
                    f"  Attack: "
                    f"{lr['n_attack']:,}",
                    f"="*50,
                ]

            if lr.get("atk_counts"):
                report_lines.append(
                    "Attack Types:")
                for k,v in sorted(
                        lr["atk_counts"].items(),
                        key=lambda x: -x[1]):
                    report_lines.append(
                        f"  {k}: {v:,}")
                report_lines.append("="*50)

            if not susp.empty:
                report_lines.append(
                    "Top Suspicious IPs:")
                for _, row in susp.head(
                        10).iterrows():
                    report_lines.append(
                        f"  {row['IP']} — "
                        f"Risk: {row['Risk Level']}"
                        f" | Requests: "
                        f"{row['Requests']}"
                        f" | Failures: "
                        f"{row['Failures']}")

            report_text = "\n".join(report_lines)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ تحميل التقرير TXT",
                    report_text,
                    f"log_report_"
                    f"{datetime.now():%Y%m%d_%H%M}"
                    f".txt",
                    "text/plain",
                    use_container_width=True,
                    key="dl_log_report")
            with c2:
                if not susp.empty:
                    st.download_button(
                        "⬇️ تحميل الـ IPs CSV",
                        susp.to_csv(index=False),
                        f"suspicious_ips_"
                        f"{datetime.now():%Y%m%d_%H%M}"
                        f".csv",
                        "text/csv",
                        use_container_width=True,
                        key="dl_log_ips")


# ══════════════════════════════════════════════════════════════
# Tab 5 — كيف يعمل
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🤖 كيف يعمل النظام")

    st.markdown("""
### 🧠 الفكرة الجوهرية — v4 Universal

بدل أن يتعلم النموذج أسماء الأعمدة مثل
`src_ip_bytes` أو `flow_duration`،
يتعلم **السلوك المجرد** للهجمات:
- كيف تبدو حركة الـ DDoS؟
- كيف يختلف الـ scanning عن الـ botnet؟
- ما هو نمط الـ exfiltration؟

هذا يجعله يعمل على **أي داتاست شبكية وأي Log**.
    """)

    st.markdown("### 🔄 Pipeline — CSV Analysis")
    st.code("""
ارفع أي CSV (TON-IoT, CIC-IoT, UNSW, ...)
         ↓
Agent 1 — Universal Parser
  يكتشف نوع الداتاست + يوحّد الـ schema
         ↓
Agent 2 — Abstract Feature Extractor
  18 behavioral features مستقلة
         ↓
Agent 3 — FT-Transformer v4
  Level 1: benign/attack
  Level 2: ddos/dos/recon/...
         ↓
Agent 4 — Explainability Engine
  SHAP + Training Curves
    """)

    st.markdown(
        "### 🔄 Pipeline — Log File Analysis")
    st.code("""
ارفع أي Log (SSH/Apache/Firewall/Windows/...)
         ↓
Log Type Detector
  يكتشف النوع من patterns تلقائياً
         ↓
Universal Log Parser
  SSH/Apache/Windows/Firewall/DNS/Generic
         ↓
Log → Abstract Features
  18 features من الأحداث
         ↓
FT-Transformer v4
  يصنّف كل حدث
         ↓
Security Report:
  Top IPs + Attack Patterns + Timeline
    """)

    st.markdown("### 🎯 الـ 18 Abstract Features")
    features_info = {
        "Volume"  : ["total_bytes",
                     "bytes_per_packet"],
        "Ratio"   : ["packet_ratio","byte_ratio",
                     "flow_asymmetry"],
        "Timing"  : ["duration_log","packet_rate",
                     "inter_arrival_var"],
        "Ports"   : ["src_port_risk",
                     "dst_port_risk",
                     "port_entropy"],
        "Protocol": ["protocol_encoded",
                     "tcp_flag_pattern"],
        "Behavior": ["connection_freq",
                     "failed_conn_ratio",
                     "payload_entropy",
                     "session_uniqueness",
                     "burstiness"],
    }
    cols = st.columns(3)
    for i, (cat, feats) in enumerate(
            features_info.items()):
        with cols[i % 3]:
            st.markdown(f"**{cat}**")
            for f in feats:
                st.write(f"  • {f}")

    st.markdown("### 📊 نتائج التدريب")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", "93.21%")
    with c2:
        st.metric("Weighted F1", "93%")
    with c3:
        st.metric("Macro F1", "82%")
    with c4:
        st.metric("Datasets", "5")


# ══════════════════════════════════════════════════════════════
# Tab 6 — عن المشروع
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📋 عن المشروع")

    st.markdown("""
### 🛡️ AI Agents IDS v4 — Universal

نظام كشف تسلل شبكي من الجيل الرابع،
يعمل على **أي داتاست شبكية وأي Log File**
بدون تكوين يدوي.

---

**الإصدارات:**
- **v3** — يعمل على TON-IoT فقط (97.8%)
- **v4** — يعمل على أي مصدر بيانات (93.2%)

---

**المميزات الجديدة في v4:**
- ✅ Universal Dataset Detection
- ✅ 18 Abstract Behavioral Features
- ✅ Hierarchical Classification (L1+L2)
- ✅ Trained on 5 Different Datasets
- ✅ Real SHAP from Actual Data
- ✅ **Log File Analysis **
  - SSH / Apache / Nginx
  - Windows Events / Firewall
  - DNS / Syslog / Generic

---

**الداتاسيت المستخدمة في التدريب:**
- TON-IoT (2023)
- CIC-IoT (2023)
- UNSW-NB15
- Bot-IoT
- CICIDS2017
    """)

    st.markdown("---")
    footer = (
        '<div style="text-align:center;'
        'padding:1rem;background:linear-gradient'
        '(135deg,#0a0a1a,#1a0a2e);'
        'border-radius:10px;'
        'border:1px solid #00d4ff33;">'
        '<p style="color:#00d4ff;font-size:1.2rem;'
        'font-weight:bold;margin:0;">'
        '🛡️ AI Agents IDS v4 Universal</p>'
        '<p style="color:#a0aec0;margin:0.3rem 0;">'
        '© 2025 <strong style="color:white;">'
        'Muaz Al-Soufi</strong></p>'
        '<p style="color:#a0aec0;margin:0.3rem 0;">'
        '<a href="https://github.com/Muoz22/'
        'ids-network-analyzer-v4" '
        'style="color:#00d4ff;">GitHub v4</a>'
        ' &nbsp;|&nbsp; '
        '<a href="https://scholar.google.com/'
        'citations?user=J35vcAIAAAAJ&hl=en" '
        'style="color:#00d4ff;">'
        'Google Scholar</a>'
        ' &nbsp;|&nbsp; '
        '<a href="https://www.researchgate.net'
        '/profile/Muaadh-Alsoufi" '
        'style="color:#00d4ff;">'
        'ResearchGate</a></p>'
        '<p style="color:#666;font-size:0.8rem;'
        'margin:0.3rem 0;">'
        'Built with ❤️ using Streamlit · '
        'ONNX Runtime · FT-Transformer · '
        'Abstract Behavioral Features · '
        'Universal Log Parser</p>'
        '</div>')
    st.markdown(footer, unsafe_allow_html=True)
