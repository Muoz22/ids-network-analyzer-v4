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
.metric-card {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px; padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size:2rem; font-weight:bold; margin:0;
}
.metric-label {
    color:#a0aec0; font-size:0.85rem;
}
.level-badge {
    display:inline-block; padding:0.3rem 0.8rem;
    border-radius:20px; font-weight:bold;
    font-size:0.85rem;
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Agents IDS v4</h1>
    <p>Universal Network Intrusion Detection —
    Powered by FT-Transformer | 5 Datasets |
    18 Abstract Features</p>
    <p style="color:#666;font-size:0.8rem;
    margin-top:0.5rem;">
    © 2025 Muaz Al-Soufi |
    <a href="https://github.com/Muoz22/
ids-network-analyzer-v4"
    style="color:#00d4ff;">GitHub v4</a>
    &nbsp;|&nbsp;
    <a href="https://github.com/Muoz22/
ids-network-analyzer"
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


# ── 5 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 تحليل الشبكة",
    "📊 تقرير تفصيلي",
    "🔬 قابلية التفسير",
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
            type=["csv"])
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

            # امسح النتائج القديمة
            for k in ["results_v4",
                      "plot_bytes_v4",
                      "exp_bytes_v4"]:
                if k in st.session_state:
                    del st.session_state[k]

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner(
                    "🔮 جاري التحليل..."):

                from inference_v4 import (
                    run_inference_v4,
                    make_plots_v4,
                    make_explainability_v4)

                if not models_global:
                    st.error(
                        "❌ النماذج غير محملة")
                    st.stop()

                status.text(
                    "🤖 تشغيل FT-Transformer v4...")
                progress.progress(20)

                results = run_inference_v4(
                    df, models_global,
                    ft_unk_thr=threshold)

                status.text(
                    "📊 إنتاج الرسوم...")
                progress.progress(50)

                with tempfile.TemporaryDirectory(
                ) as tmp:
                    plot_paths = make_plots_v4(
                        results,
                        out_dir=tmp,
                        models_ref=models_global)
                    plot_bytes = []
                    for title, path in \
                            plot_paths:
                        try:
                            with open(
                                    path,"rb") as f:
                                plot_bytes.append(
                                    (title,
                                     f.read()))
                        except Exception:
                            pass

                status.text(
                    "🔬 حساب قابلية التفسير...")
                progress.progress(80)

                exp_bytes = []
                with tempfile.TemporaryDirectory(
                ) as tmp2:
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

                # ── النتائج ───────────────────
                st.markdown("---")
                st.markdown(
                    "## 🏆 نتائج التحليل")

                # Dataset detected badge
                ds = results.get(
                    "detected_ds", "AUTO")
                st.markdown(
                    f'**Dataset detected:** '
                    f'<span class="dataset-badge">'
                    f'{ds}</span>',
                    unsafe_allow_html=True)

                # Level 1 + Level 2
                total   = results["n_samples"]
                benign  = results["n_benign"]
                attacks = results["n_attacks"]
                unknown = results["n_unknown"]
                m       = results["metrics"]

                # Level 1
                st.markdown(
                    "### 🎯 Level 1 — "
                    "Benign / Attack")
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric(
                        "إجمالي",
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
                    l1a = m["l1_accuracy"]*100
                    l1f = m["l1_f1"]*100
                    st.markdown(
                        f"**Level 1 Accuracy:** "
                        f"`{l1a:.2f}%`  |  "
                        f"**F1:** `{l1f:.2f}%`")

                # Level 2
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
                             80,60),
                            ]):
                        with col:
                            cl = (
                                "status-excellent"
                                if val>hi else
                                "status-good"
                                if val>lo else
                                "status-warning")
                            st.markdown(
                                f'<p class='
                                f'"metric-value '
                                f'{cl}">'
                                f'{val:.2f}%</p>'
                                f'<p class='
                                f'"metric-label">'
                                f'{name}</p>',
                                unsafe_allow_html
                                =True)

                # Attack types
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
                        total * 100
                    ).round(2)
                    st.dataframe(
                        atk_df,
                        use_container_width=True)

                # Plots
                if plot_bytes:
                    st.markdown(
                        "### 📊 الرسوم البيانية")
                    for i in range(
                            0, len(plot_bytes),
                            2):
                        cols = st.columns(2)
                        for j, (title, img) \
                                in enumerate(
                                plot_bytes[i:i+2]
                                ):
                            with cols[j]:
                                st.markdown(
                                    f"**{title}**")
                                st.image(
                                    img,
                                    use_column_width
                                    =True)

                if "report" in m:
                    with st.expander(
                            "📋 Classification "
                            "Report"):
                        st.code(m["report"])

                st.markdown(
                    "### 💾 تحميل النتائج")
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
        results    = st.session_state[
            "results_v4"]
        plot_bytes = st.session_state[
            "plot_bytes_v4"]
        m          = results["metrics"]
        total      = results["n_samples"]
        benign     = results["n_benign"]
        attacks    = results["n_attacks"]
        unknown    = results["n_unknown"]
        ds         = results.get(
            "detected_ds", "AUTO")

        st.markdown(
            f'**Dataset:** '
            f'<span class="dataset-badge">'
            f'{ds}</span>  '
            f'**Samples:** {total:,}',
            unsafe_allow_html=True)

        # Level 1
        st.markdown(
            "### 🎯 Level 1 — Benign/Attack")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric("✅ ALLOW",
                      f"{benign:,}",
                      f"{100*benign/total:.1f}%")
        with c2:
            st.metric("🚫 BLOCK",
                      f"{attacks:,}",
                      f"{100*attacks/total:.1f}%",
                      delta_color="inverse")
        with c3:
            st.metric(
                "⚠️ QUARANTINE",
                f"{unknown:,}",
                f"{100*unknown/total:.1f}%",
                delta_color="inverse")

        if "l1_accuracy" in m:
            st.success(
                f"Level 1 — "
                f"Accuracy: "
                f"{m['l1_accuracy']*100:.2f}%  |  "
                f"F1: {m['l1_f1']*100:.2f}%")

        # Decision + Attack
        st.markdown("---")
        st.markdown("### 🎯 Decision Analysis")
        pie_imgs = [
            (t,img) for t,img in plot_bytes
            if "Distribution" in t]
        atk_imgs = [
            (t,img) for t,img in plot_bytes
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

        # Level 2 Performance
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
                     m["macro_f1"]*100,80,60),
                    ]):
                with col:
                    cl = (
                        "status-excellent"
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

            # CM + Training Curves
            conf_imgs  = [
                (t,img) for t,img in plot_bytes
                if "Confusion" in t]
            train_imgs = [
                (t,img) for t,img in plot_bytes
                if "Training" in t]
            if conf_imgs or train_imgs:
                pf1, pf2 = st.columns(2)
                with pf1:
                    if conf_imgs:
                        st.markdown(
                            "**Confusion Matrix**")
                        st.image(
                            conf_imgs[0][1],
                            use_column_width=True)
                with pf2:
                    if train_imgs:
                        st.markdown(
                            "**Training Curves**")
                        st.image(
                            train_imgs[0][1],
                            use_column_width=True)

            f1_imgs = [
                (t,img) for t,img in plot_bytes
                if "F1" in t]
            if f1_imgs:
                st.markdown("**Per-Class F1**")
                st.image(f1_imgs[0][1],
                         use_column_width=True)

            if "report" in m:
                st.markdown("---")
                st.markdown(
                    "### 📋 Classification "
                    "Report")
                st.code(m["report"])

        # Confidence
        st.markdown("---")
        st.markdown(
            "### 🎚️ Confidence Analysis")
        conf_imgs = [
            (t,img) for t,img in plot_bytes
            if t == "Confidence"]
        if conf_imgs:
            st.image(conf_imgs[0][1],
                     use_column_width=True)

        # Drift
        st.markdown("---")
        st.markdown("### 🌊 Drift Analysis")
        drift_imgs = [
            (t,img) for t,img in plot_bytes
            if "Drift" in t]
        if drift_imgs:
            st.image(drift_imgs[0][1],
                     use_column_width=True)

        # Dashboard
        st.markdown("---")
        dash_imgs = [
            (t,img) for t,img in plot_bytes
            if "Dashboard" in t]
        if dash_imgs:
            st.markdown(
                "### 🗂️ Summary Dashboard")
            st.image(dash_imgs[0][1],
                     use_column_width=True)

        # All plots
        st.markdown("---")
        st.markdown(
            "### 🖼️ كل الرسوم البيانية")
        for i in range(0, len(plot_bytes), 2):
            cols = st.columns(2)
            for j, (title, img) in enumerate(
                    plot_bytes[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(
                        img,
                        use_column_width=True)

        # Download
        st.markdown("---")
        result_df = pd.DataFrame({
            "prediction_l2":
                results["y_pred_l2"],
            "prediction_l1":
                results["y_pred_l1"],
            "confidence": results["y_conf"],
            "is_unknown": results["y_unknown"],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ تحميل النتائج CSV",
                result_df.to_csv(index=False),
                f"ids_v4_"
                f"{datetime.now():%Y%m%d_%H%M}"
                f".csv",
                "text/csv",
                use_container_width=True,
                key="dl_tab2")
        with c2:
            if "report" in m:
                st.download_button(
                    "⬇️ تحميل التقرير TXT",
                    m["report"],
                    f"report_v4_"
                    f"{datetime.now():%Y%m%d_%H%M}"
                    f".txt",
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
        exp_bytes = st.session_state[
            "exp_bytes_v4"]
        results   = st.session_state[
            "results_v4"]
        ds        = results.get(
            "detected_ds", "AUTO")

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
            st.warning(
                "⚠️ لم تُنتج رسوم التفسير")
        else:
            # Training Curves أولاً
            train_exp = [
                (t,img) for t,img in exp_bytes
                if "Training" in t]
            if train_exp:
                st.markdown(
                    "### 📈 Training Curves "
                    "— FT-Transformer v4")
                st.markdown(
                    "*النموذج تدرّب على 5 "
                    "داتاسيت مختلفة*")
                st.image(
                    train_exp[0][1],
                    use_column_width=True)

            st.markdown("---")

            # SHAP Summary + Waterfall
            shap_s = [
                (t,img) for t,img in exp_bytes
                if "SHAP Summary" in t]
            shap_w = [
                (t,img) for t,img in exp_bytes
                if "Waterfall" in t]

            if shap_s or shap_w:
                st.markdown(
                    "### 🧠 SHAP Analysis")
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
                        st.image(
                            shap_s[0][1],
                            use_column_width=True)
                with sc2:
                    if shap_w:
                        st.markdown(
                            "**SHAP Waterfall "
                            "— Cumulative**")
                        st.image(
                            shap_w[0][1],
                            use_column_width=True)

            st.markdown("---")

            # باقي الرسوم
            other_exp = [
                (t,img) for t,img in exp_bytes
                if "Training" not in t
                and "SHAP" not in t]
            if other_exp:
                st.markdown(
                    "### 📊 تحليل إضافي")
                for i in range(
                        0, len(other_exp), 2):
                    cols = st.columns(2)
                    for j, (title, img) \
                            in enumerate(
                            other_exp[i:i+2]):
                        with cols[j]:
                            st.markdown(
                                f"**{title}**")
                            st.image(
                                img,
                                use_column_width
                                =True)

        # Feature importance table
        st.markdown("---")
        st.markdown(
            "### 📋 Feature Importance Table")
        feat_names = results.get(
            "feat_names", [])
        X          = results["X_final"]
        if len(feat_names) > 0:
            imp = np.abs(X).mean(axis=0)
            std = np.abs(X).std(axis=0)
            idx = np.argsort(imp)[::-1]
            feat_df = pd.DataFrame({
                "Feature": [
                    feat_names[i]
                    for i in idx],
                "Importance": [
                    f"{imp[i]:.4f}"
                    for i in idx],
                "Std": [
                    f"{std[i]:.4f}"
                    for i in idx],
            })
            st.dataframe(
                feat_df,
                use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Tab 4 — كيف يعمل
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🤖 كيف يعمل النظام")

    st.markdown("""
### 🧠 الفكرة الجوهرية — v4 Universal

بدل أن يتعلم النموذج أسماء الأعمدة مثل
`src_ip_bytes` أو `flow_duration`،
يتعلم **السلوك المجرد** للهجمات:
- كيف تبدو حركة الـ DDoS؟
- كيف يختلف الـ scanning عن الـ botnet؟
- ما هو نمط الـ exfiltration؟

هذا يجعله يعمل على **أي داتاست شبكية**.
    """)

    st.markdown("### 🔄 Pipeline الكامل")
    st.code("""
ارفع أي CSV (TON-IoT, CIC-IoT, UNSW, ...)
         ↓
Agent 1 — Universal Parser
  • يكتشف نوع الداتاست تلقائياً
  • يوحّد الـ schema
  • يوحّد الـ labels

         ↓
Agent 2 — Abstract Feature Extractor
  • يحسب 18 behavioral feature
  • مستقلة عن أسماء الأعمدة
  • تعمل على أي داتاست

         ↓
Agent 3 — FT-Transformer v4
  • Level 1: benign / attack
  • Level 2: ddos/dos/recon/...
  • تدرّب على 5 داتاسيت مختلفة

         ↓
Agent 4 — Explainability Engine
  • SHAP values من بياناتك الفعلية
  • Training Curves
  • Confidence Analysis

         ↓
النتائج: ALLOW / BLOCK / QUARANTINE
    """)

    st.markdown("### 🎯 الـ 18 Abstract Features")

    features_info = {
        "Volume"  : [
            "total_bytes",
            "bytes_per_packet"],
        "Ratio"   : [
            "packet_ratio","byte_ratio",
            "flow_asymmetry"],
        "Timing"  : [
            "duration_log","packet_rate",
            "inter_arrival_var"],
        "Ports"   : [
            "src_port_risk","dst_port_risk",
            "port_entropy"],
        "Protocol": [
            "protocol_encoded",
            "tcp_flag_pattern"],
        "Behavior": [
            "connection_freq",
            "failed_conn_ratio",
            "payload_entropy",
            "session_uniqueness",
            "burstiness"],
    }

    cols = st.columns(3)
    for i, (category, feats) in enumerate(
            features_info.items()):
        with cols[i % 3]:
            st.markdown(
                f"**{category}**")
            for f in feats:
                st.write(f"  • {f}")

    st.markdown("### 📊 نتائج التدريب")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "93.21%")
    with col2:
        st.metric("Weighted F1", "93%")
    with col3:
        st.metric("Macro F1", "82%")
    with col4:
        st.metric("Datasets", "5")


# ══════════════════════════════════════════════════════════════
# Tab 5 — عن المشروع
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📋 عن المشروع")

    st.markdown("""
### 🛡️ AI Agents IDS v4 — Universal

نظام كشف تسلل شبكي من الجيل الرابع،
يعمل على **أي داتاست شبكية** بدون تكوين يدوي.

---

**الإصدارات:**
- **v3** — يعمل على TON-IoT فقط (97.8%)
- **v4** — يعمل على أي داتاست (93.2%)

---

**المميزات الجديدة في v4:**
- ✅ Universal Dataset Detection
- ✅ 18 Abstract Behavioral Features
- ✅ Hierarchical Classification (L1+L2)
- ✅ Trained on 5 Different Datasets
- ✅ Real SHAP from Actual Data
- ✅ Dedicated Explainability Tab

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
        '<p style="color:#00d4ff;'
        'font-size:1.2rem;'
        'font-weight:bold;margin:0;">'
        '🛡️ AI Agents IDS v4 Universal</p>'
        '<p style="color:#a0aec0;'
        'margin:0.3rem 0;">'
        '© 2025 <strong style="color:white;">'
        'Muaz Al-Soufi</strong></p>'
        '<p style="color:#a0aec0;'
        'margin:0.3rem 0;">'
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
        '<p style="color:#666;'
        'font-size:0.8rem;margin:0.3rem 0;">'
        'Built with ❤️ using Streamlit · '
        'ONNX Runtime · FT-Transformer · '
        'Abstract Behavioral Features</p>'
        '</div>')
    st.markdown(footer, unsafe_allow_html=True)
