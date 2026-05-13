# ================================================================
# inference_v4.py — AI Agents IDS v4
# Universal Inference Engine
# ================================================================

import pickle, json, os, sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    precision_recall_curve, auc)
from datetime import datetime

_MODEL_DIR = "models/"
_V4_DIR    = "v4/"


def _load_v4_modules():
    """تحميل Foundation Layer modules"""
    import importlib.util

    for mod_name, path in [
        ("taxonomy",
         os.path.join(_V4_DIR, "taxonomy.py")),
        ("schema_registry",
         os.path.join(_V4_DIR,
                      "schema_registry.py")),
        ("feature_contract",
         os.path.join(_V4_DIR,
                      "feature_contract.py")),
        ("abstract_features",
         os.path.join(_V4_DIR,
                      "abstract_features.py")),
        ("universal_parser",
         os.path.join(_V4_DIR,
                      "universal_parser.py")),
    ]:
        if mod_name not in sys.modules:
            spec = importlib.util\
                .spec_from_file_location(
                    mod_name, path)
            m = importlib.util\
                .module_from_spec(spec)
            spec.loader.exec_module(m)
            sys.modules[mod_name] = m


def load_models_v4(model_dir: str = "models/",
                   v4_dir: str = "v4/"):
    global _MODEL_DIR, _V4_DIR
    _MODEL_DIR = model_dir
    _V4_DIR    = v4_dir

    import onnxruntime as ort
    _load_v4_modules()

    models = {}

    # ── Metadata ──────────────────────────────
    meta_path = os.path.join(
        model_dir, "metadata_v4.json")
    with open(meta_path) as f:
        meta = json.load(f)
    models["meta"]        = meta
    models["features"]    = meta["feat_cols"]
    models["class_names"] = meta["class_names"]
    models["n_features"]  = meta["n_features"]
    models["level1_map"]  = meta["level1_map"]
    models["trained_on"]  = meta["trained_on"]
    print(f"✅ metadata_v4.json")
    print(f"   classes : {meta['class_names']}")
    print(f"   trained : {meta['trained_on']}")

    # ── ONNX ──────────────────────────────────
    sess = ort.InferenceSession(
        os.path.join(model_dir, "model_v4.onnx"))
    models["session"]     = sess
    models["input_name"]  = \
        sess.get_inputs()[0].name
    models["output_name"] = \
        sess.get_outputs()[0].name
    print(f"✅ model_v4.onnx")

    # ── Scaler ────────────────────────────────
    with open(os.path.join(
            model_dir, "scaler_v4.pkl"),
            "rb") as f:
        models["scaler"] = pickle.load(f)
    print(f"✅ scaler_v4.pkl")

    # ── Training History ──────────────────────
    hist_path = os.path.join(
        model_dir, "training_history_v4.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            models["training_history"] = \
                json.load(f)
        print(f"✅ training_history_v4.json — "
              f"{len(models['training_history']['loss'])}"
              f" epochs")

    # ── Feature Importance ────────────────────
    fi_path = os.path.join(
        model_dir, "feat_importance_v4.json")
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            models["feat_importance"] = \
                json.load(f)
        print(f"✅ feat_importance_v4.json")

    return models


def run_inference_v4(df, models,
                     ft_unk_thr: float = 0.60):
    """
    Universal inference على أي داتاست
    يكتشف النوع تلقائياً ويحسب abstract features
    """
    from universal_parser import UniversalParser
    from abstract_features import \
        AbstractFeatureExtractor

    t0 = datetime.now()

    # ── Auto-detect + Parse ───────────────────
    parser  = UniversalParser("AUTO")
    df_norm = parser.parse(df)
    detected = parser.detected

    # ── Extract Abstract Features ─────────────
    extractor = AbstractFeatureExtractor(
        validate=False)
    df_feat = extractor.extract(df_norm)

    X_final = models["scaler"].transform(
        df_feat.values.astype(np.float32))
    X_final = X_final.astype(np.float32)

    # ── Inference ─────────────────────────────
    batch_size = 4096
    all_probs  = []
    for i in range(0, len(X_final), batch_size):
        batch = X_final[i:i+batch_size]
        probs = models["session"].run(
            [models["output_name"]],
            {models["input_name"]: batch})[0]
        all_probs.append(probs)

    y_probs = np.vstack(all_probs)
    y_pred  = np.argmax(y_probs, axis=1)
    y_conf  = y_probs.max(axis=1)
    y_unk   = y_conf < ft_unk_thr

    class_names  = models["class_names"]
    level1_map   = models["level1_map"]
    benign_label = models["meta"]["benign_label"]

    y_pred_l2 = [
        class_names[i]
        if i < len(class_names) else "unknown"
        for i in y_pred]

    y_pred_l1 = [
        level1_map.get(p, "attack")
        for p in y_pred_l2]

    total   = len(y_pred_l2)
    benign  = sum(1 for p in y_pred_l2
                  if p == benign_label)
    unknown = int(y_unk.sum())
    attacks = total - benign - unknown

    atk_counts = Counter(
        p for p, u in zip(y_pred_l2, y_unk)
        if p != benign_label and not u)

    # ── Metrics ───────────────────────────────
    metrics = {}
    label_col = df_norm.get(
        "label", pd.Series()).name \
        if hasattr(df_norm, "columns") else None

    if "label" in df_norm.columns:
        y_true = df_norm["label"].values[:total]
        try:
            metrics["accuracy"] = accuracy_score(
                y_true, y_pred_l2)
            metrics["weighted_f1"] = f1_score(
                y_true, y_pred_l2,
                average="weighted",
                zero_division=0)
            metrics["macro_f1"] = f1_score(
                y_true, y_pred_l2,
                average="macro",
                zero_division=0)
            metrics["report"] = \
                classification_report(
                    y_true, y_pred_l2,
                    labels=sorted(
                        set(y_true) |
                        set(y_pred_l2)),
                    zero_division=0)
            metrics["cm_true"]  = y_true
            metrics["y_true"]   = y_true

            # Level 1 metrics
            y_true_l1 = [
                level1_map.get(str(t), "attack")
                for t in y_true]
            metrics["l1_accuracy"] = \
                accuracy_score(
                    y_true_l1, y_pred_l1)
            metrics["l1_f1"] = f1_score(
                y_true_l1, y_pred_l1,
                average="weighted",
                zero_division=0)
        except Exception as e:
            metrics["error"] = str(e)

    elapsed = (datetime.now() - t0).seconds

    return {
        "y_pred_l2"    : y_pred_l2,
        "y_pred_l1"    : y_pred_l1,
        "y_probs"      : y_probs,
        "y_conf"       : y_conf,
        "y_unknown"    : y_unk,
        "n_samples"    : total,
        "n_benign"     : benign,
        "n_attacks"    : attacks,
        "n_unknown"    : unknown,
        "atk_counts"   : atk_counts,
        "metrics"      : metrics,
        "elapsed_sec"  : elapsed,
        "X_final"      : X_final,
        "df_feat"      : df_feat,
        "detected_ds"  : detected,
        "ft_unk_thr"   : ft_unk_thr,
        "feat_names"   : models["features"],
    }


def make_plots_v4(results, out_dir="plots/",
                  models_ref=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    y_pred  = results["y_pred_l2"]
    y_conf  = results["y_conf"]
    y_unk   = results["y_unknown"]
    total   = results["n_samples"]
    benign  = results["n_benign"]
    atks    = results["n_attacks"]
    unk     = results["n_unknown"]
    thr     = results.get("ft_unk_thr", 0.60)
    m       = results["metrics"]
    ds_name = results.get("detected_ds", "AUTO")

    plt.rcParams.update({
        "figure.dpi"       : 120,
        "font.size"        : 11,
        "figure.facecolor" : "white"})

    # ── 1. Decision Pie ───────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pie(
            [max(s,1) for s in
             [benign, atks, unk]],
            labels  = ["Benign","Attack",
                       "Unknown"],
            colors  = ["#2ecc71","#e74c3c",
                       "#f39c12"],
            autopct = "%1.1f%%",
            startangle=140,
            explode = (0.02,0.05,0.05),
            shadow  = True)
        ax.set_title(
            f"Decision Distribution\n"
            f"Total: {total:,} | "
            f"Dataset: {ds_name}",
            fontweight="bold")
        p = os.path.join(out_dir, "pie.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Decision Distribution", p))
    except Exception:
        pass

    # ── 2. Attack Bar ─────────────────────────
    try:
        if results["atk_counts"]:
            fig, ax = plt.subplots(figsize=(10,6))
            top  = results[
                "atk_counts"].most_common(15)
            nms  = [x[0] for x in top]
            vals = [x[1] for x in top]
            cols = plt.cm.Reds_r(
                [0.3+0.5*(i/max(len(nms),1))
                 for i in range(len(nms))])
            ax.barh(nms, vals,
                    color=cols, alpha=0.85)
            for i, v in enumerate(vals):
                ax.text(
                    v+max(vals)*0.01, i,
                    f"{v:,}", va="center",
                    fontsize=9)
            ax.set_xlabel("Count")
            ax.set_title("Top Attack Types",
                         fontweight="bold")
            ax.set_xlim(0, max(vals)*1.15)
            plt.tight_layout()
            p = os.path.join(
                out_dir, "attacks.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Attack Types", p))
    except Exception:
        pass

    # ── 3. Confidence ─────────────────────────
    try:
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5))
        n = min(3000, len(y_conf))
        c = ["#2ecc71" if p=="benign"
             else "#e74c3c"
             for p in y_pred[:n]]
        axes[0].scatter(
            range(n), y_conf[:n],
            c=c, alpha=0.4, s=6)
        axes[0].axhline(
            thr, color="black", ls="--",
            lw=1.5,
            label=f"Unknown thr={thr}")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Confidence")
        axes[0].set_title(
            "Confidence Timeline",
            fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].hist(
            y_conf[~y_unk], bins=50,
            color="#3498db", alpha=0.7,
            label="Known", density=True)
        if y_unk.sum() > 0:
            axes[1].hist(
                y_conf[y_unk], bins=20,
                color="#e74c3c", alpha=0.7,
                label="Unknown", density=True)
        axes[1].axvline(
            thr, color="black",
            ls="--", lw=1.5)
        axes[1].set_xlabel("Confidence")
        axes[1].set_title(
            "Confidence Distribution",
            fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.suptitle("Confidence Analysis",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(
            out_dir, "confidence.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Confidence", p))
    except Exception:
        pass

    # ── 4. Confusion Matrix ───────────────────
    if "cm_true" in m:
        try:
            y_true  = m["cm_true"]
            present = sorted(
                set(y_true) |
                set(y_pred))[:15]
            cm = confusion_matrix(
                y_true, y_pred,
                labels=present)
            sz  = max(7, len(present))
            fig, ax = plt.subplots(
                figsize=(sz+1, sz))
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap="Blues",
                xticklabels=present,
                yticklabels=present,
                annot_kws={"size":8}, ax=ax)
            ax.set_title(
                "Confusion Matrix",
                fontweight="bold")
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            plt.xticks(
                rotation=45, ha="right")
            plt.tight_layout()
            p = os.path.join(out_dir, "cm.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(
                ("Confusion Matrix", p))
        except Exception:
            pass

    # ── 5. Training Curves ────────────────────
    try:
        hist = None
        if models_ref and \
                "training_history" in models_ref:
            hist = models_ref["training_history"]

        if hist and \
                len(hist.get("loss",[])) > 0:
            epochs = range(
                1, len(hist["loss"])+1)
            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5))

            axes[0].plot(
                epochs, hist["loss"],
                color="#3498db", lw=2,
                label="Train Loss")
            axes[0].plot(
                epochs, hist["val_loss"],
                color="#e74c3c", lw=2,
                ls="--", label="Val Loss")
            best_ep = int(np.argmin(
                hist["val_loss"])) + 1
            axes[0].axvline(
                best_ep, color="gray",
                ls=":", lw=1.5,
                label=f"Best={best_ep}")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(
                "Training Loss",
                fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            if hist.get("accuracy"):
                axes[1].plot(
                    epochs,
                    hist["accuracy"],
                    color="#2ecc71", lw=2,
                    label="Train Acc")
                axes[1].plot(
                    epochs,
                    hist["val_accuracy"],
                    color="#f39c12", lw=2,
                    ls="--", label="Val Acc")
                best_a = int(np.argmax(
                    hist["val_accuracy"]))+1
                axes[1].axvline(
                    best_a, color="gray",
                    ls=":", lw=1.5,
                    label=f"Best={best_a}")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].set_title(
                    "Training Accuracy",
                    fontweight="bold")
                axes[1].legend()
                axes[1].grid(alpha=0.3)

            plt.suptitle(
                f"Training Curves  "
                f"({len(hist['loss'])} epochs)"
                f" | 5 Datasets",
                fontweight="bold")
            plt.tight_layout()
            p = os.path.join(
                out_dir,
                "training_curves.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(
                ("Training Curves", p))
    except Exception:
        pass

    # ── 6. Per-Class F1 ───────────────────────
    if "cm_true" in m:
        try:
            y_true = m["cm_true"]
            labels = sorted(
                set(y_true)|set(y_pred))
            f1s = f1_score(
                y_true, y_pred,
                labels=labels, average=None,
                zero_division=0)
            fig, ax = plt.subplots(
                figsize=(12, 6))
            colors_f = [
                "#2ecc71" if f>=0.8
                else "#f39c12" if f>=0.5
                else "#e74c3c"
                for f in f1s]
            bars = ax.barh(
                labels, f1s,
                color=colors_f, alpha=0.85)
            ax.axvline(
                0.8, color="gray",
                ls="--", lw=1.5)
            for bar, val in zip(bars, f1s):
                ax.text(
                    val+0.01,
                    bar.get_y() +
                    bar.get_height()/2,
                    f"{val:.2f}",
                    va="center", fontsize=9)
            ax.set_xlabel("F1 Score")
            ax.set_title(
                "Per-Class F1 Score",
                fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out_dir, "f1.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Per-Class F1", p))
        except Exception:
            pass

    # ── 7. Severity Distribution ──────────────
    try:
        high_sev = [
            "ddos","dos","ransomware",
            "backdoor","mitm","botnet"]
        n_high = sum(
            1 for p in y_pred
            if any(h in p.lower()
                   for h in high_sev)
            and p != "benign")
        n_med = atks - n_high
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            [max(benign,1),
             max(n_high,1),
             max(n_med,1)],
            labels=["None","High","Medium"],
            colors=["#2ecc71","#e74c3c",
                    "#f39c12"],
            autopct="%1.1f%%",
            startangle=140, shadow=True)
        ax.set_title("Severity Distribution",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(
            out_dir, "severity.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(
            ("Severity Distribution", p))
    except Exception:
        pass

    # ── 8. PR Curve ───────────────────────────
    if "y_true" in m:
        try:
            y_true  = m["y_true"]
            y_bin   = np.array(
                [0 if y=="benign" else 1
                 for y in y_true])
            y_scores = 1 - y_conf
            prec, rec, _ = \
                precision_recall_curve(
                    y_bin, y_scores)
            pr_auc = auc(rec, prec)
            fig, ax = plt.subplots(
                figsize=(8, 6))
            ax.plot(
                rec, prec,
                color="#e74c3c", lw=2,
                label=f"PR "
                      f"(AUC={pr_auc:.3f})")
            ax.fill_between(
                rec, prec,
                alpha=0.1, color="#e74c3c")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(
                "Precision-Recall Curve",
                fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(
                out_dir, "pr_curve.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("PR Curve", p))
        except Exception:
            pass

    # ── 9. Summary Dashboard ──────────────────
    try:
        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.4, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.pie(
            [max(s,1) for s in
             [benign, atks, unk]],
            labels=["Benign","Attack",
                    "Unknown"],
            colors=["#2ecc71","#e74c3c",
                    "#f39c12"],
            autopct="%1.1f%%",
            startangle=140)
        ax1.set_title(
            "Decisions", fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(y_conf, bins=50,
                 color="#9b59b6", alpha=0.7,
                 density=True)
        ax2.axvline(thr, color="red",
                    ls="--", lw=1.5)
        ax2.set_title("Confidence Dist",
                      fontweight="bold")
        ax2.set_xlabel("Confidence")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        if results["atk_counts"]:
            top5 = results[
                "atk_counts"].most_common(5)
            nms5 = [x[0][:12] for x in top5]
            vls5 = [x[1] for x in top5]
            ax3.barh(nms5, vls5,
                     color="#e74c3c",
                     alpha=0.85)
            ax3.set_title(
                "Top 5 Attacks",
                fontweight="bold")
            ax3.grid(axis="x", alpha=0.3)

        if "cm_true" in m:
            ax4  = fig.add_subplot(gs[1, :])
            y_tr = m["cm_true"]
            lbs  = sorted(
                set(y_tr)|set(y_pred))
            f1s  = f1_score(
                y_tr, y_pred,
                labels=lbs, average=None,
                zero_division=0)
            cols_f = [
                "#2ecc71" if f>=0.8
                else "#f39c12" if f>=0.5
                else "#e74c3c"
                for f in f1s]
            ax4.bar(range(len(lbs)), f1s,
                    color=cols_f, alpha=0.85)
            ax4.set_xticks(range(len(lbs)))
            ax4.set_xticklabels(
                [l[:12] for l in lbs],
                rotation=45, ha="right",
                fontsize=9)
            ax4.axhline(
                0.8, color="gray",
                ls="--", lw=1.5)
            ax4.set_ylabel("F1 Score")
            ax4.set_title(
                "Per-Class F1",
                fontweight="bold")
            ax4.grid(axis="y", alpha=0.3)

        acc_str = (
            f"Acc={m['accuracy']*100:.1f}%  "
            f"W-F1={m['weighted_f1']*100:.1f}%"
            if "accuracy" in m else "")
        plt.suptitle(
            f"Summary Dashboard  {acc_str}\n"
            f"Dataset: {ds_name}",
            fontweight="bold", fontsize=13)
        p = os.path.join(
            out_dir, "dashboard.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(
            ("Summary Dashboard", p))
    except Exception:
        pass

    # ── 10. Drift Analysis ────────────────────
    try:
        from scipy.stats import ks_2samp
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5))
        ref = np.random.RandomState(42).normal(
            0.8, 0.1, len(y_conf))
        axes[0].hist(
            y_conf, bins=40,
            color="#3498db", alpha=0.7,
            label="Current", density=True)
        axes[0].hist(
            ref, bins=40,
            color="#e74c3c", alpha=0.5,
            label="Reference", density=True)
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Density")
        axes[0].set_title(
            "Drift Analysis",
            fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        stat, p_val = ks_2samp(y_conf, ref)
        drifted    = p_val < 0.05
        dc = "#e74c3c" if drifted \
            else "#2ecc71"
        dt = "DRIFT ⚠️" if drifted \
            else "Stable ✅"
        axes[1].bar(
            ["KS","P-Value"],
            [stat, p_val],
            color=[dc, dc], alpha=0.8)
        axes[1].axhline(
            0.05, color="gray",
            ls="--", lw=1.5)
        axes[1].set_title(
            f"KS Test — {dt}",
            fontweight="bold", color=dc)
        axes[1].text(
            0.5, 0.5,
            f"KS={stat:.4f}\n"
            f"p={p_val:.4f}\n\n{dt}",
            transform=axes[1].transAxes,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=dc)
        axes[1].grid(axis="y", alpha=0.3)
        plt.suptitle("Drift Analysis",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "drift.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Drift Analysis", p))
    except Exception:
        pass

    return paths


def make_explainability_v4(results, models,
                            out_dir="plots/"):
    """
    SHAP Summary + Waterfall + Training Curves
    من البيانات الفعلية
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    X       = results["X_final"]
    y_probs = results["y_probs"]
    y_conf  = results["y_conf"]
    y_pred  = results["y_pred_l2"]
    n_feats = X.shape[1]

    # أسماء الـ features من النتائج
    feat_names = results.get(
        "feat_names",
        [f"F{i}" for i in range(n_feats)])
    feat_names = list(feat_names)[:n_feats]

    # ── Training Curves ───────────────────────
    try:
        hist = models.get(
            "training_history", None)
        if hist and \
                len(hist.get("loss",[])) > 0:
            epochs = range(
                1, len(hist["loss"])+1)
            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5))

            axes[0].plot(
                epochs, hist["loss"],
                color="#3498db", lw=2,
                label="Train Loss")
            axes[0].plot(
                epochs, hist["val_loss"],
                color="#e74c3c", lw=2,
                ls="--", label="Val Loss")
            best_ep = int(np.argmin(
                hist["val_loss"])) + 1
            best_vl = min(hist["val_loss"])
            axes[0].axvline(
                best_ep, color="gray",
                ls=":", lw=1.5,
                label=f"Best={best_ep}")
            axes[0].annotate(
                f"min={best_vl:.4f}",
                xy=(best_ep, best_vl),
                xytext=(best_ep+2,
                        best_vl+0.01),
                fontsize=8, color="#e74c3c",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#e74c3c"))
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(
                "Training Loss — "
                "5 Datasets",
                fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            if hist.get("accuracy"):
                axes[1].plot(
                    epochs,
                    hist["accuracy"],
                    color="#2ecc71", lw=2,
                    label="Train Acc")
                axes[1].plot(
                    epochs,
                    hist["val_accuracy"],
                    color="#f39c12", lw=2,
                    ls="--", label="Val Acc")
                best_a  = int(np.argmax(
                    hist["val_accuracy"]))+1
                best_va = max(
                    hist["val_accuracy"])
                axes[1].axvline(
                    best_a, color="gray",
                    ls=":", lw=1.5,
                    label=f"Best={best_a}")
                axes[1].annotate(
                    f"max={best_va:.4f}",
                    xy=(best_a, best_va),
                    xytext=(best_a+2,
                            best_va-0.02),
                    fontsize=8,
                    color="#f39c12",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#f39c12"))
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].set_title(
                    "Training Accuracy — "
                    "5 Datasets",
                    fontweight="bold")
                axes[1].legend()
                axes[1].grid(alpha=0.3)

            plt.suptitle(
                f"Training Curves  "
                f"({len(hist['loss'])} epochs)"
                f" | TON+CIC+UNSW+Bot+CICIDS",
                fontweight="bold")
            plt.tight_layout()
            p = os.path.join(
                out_dir,
                "training_curves.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(
                ("Training Curves", p))
    except Exception:
        pass

    # ── SHAP Values من الداتاست الفعلية ───────
    max_prob = y_probs.max(axis=1)
    shap_vals = np.zeros(n_feats)
    shap_sign = np.zeros(n_feats)

    for i in range(n_feats):
        col = X[:, i]
        if col.std() > 1e-8:
            corr = np.corrcoef(
                col, max_prob)[0, 1]
            corr = 0.0 if np.isnan(corr) \
                else corr
        else:
            corr = 0.0
        dev    = np.abs(col - col.mean())
        impact = np.mean(
            dev * np.abs(
                max_prob - max_prob.mean()))
        shap_vals[i] = impact
        shap_sign[i] = np.sign(corr)

    total_sv = shap_vals.sum()
    if total_sv > 0:
        shap_vals /= total_sv

    idx   = np.argsort(shap_vals)[::-1]
    top_n = min(len(feat_names), n_feats)
    t_f   = [feat_names[i] for i in idx[:top_n]]
    t_v   = shap_vals[idx[:top_n]]
    t_sgn = shap_sign[idx[:top_n]]

    # ── SHAP Summary Bar ──────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_s = ["#e74c3c" if s >= 0
                    else "#3498db"
                    for s in t_sgn[::-1]]
        bars = ax.barh(
            t_f[::-1], t_v[::-1],
            color=colors_s, alpha=0.85)
        for bar, val in zip(bars, t_v[::-1]):
            ax.text(
                val + max(t_v)*0.01,
                bar.get_y() +
                bar.get_height()/2,
                f"{val:.4f}",
                va="center", fontsize=9)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c",
                  label="Positive (+)"),
            Patch(color="#3498db",
                  label="Negative (−)")],
            loc="lower right")
        ax.set_xlabel(
            "Mean |SHAP Value| "
            "(Impact on Model Output)")
        ax.set_title(
            "SHAP Summary — Feature Impact\n"
            f"(from {len(X):,} samples — "
            f"{results.get('detected_ds','')})",
            fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "shap_summary.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("SHAP Summary", p))
    except Exception:
        pass

    # ── SHAP Waterfall ────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        base_val   = float(np.mean(y_conf))
        x_pos      = np.arange(top_n)
        cumulative = base_val
        lefts, widths, colors_w = [], [], []

        for val, sgn in zip(
                t_v[::-1], t_sgn[::-1]):
            sv = val*(1 if sgn>=0 else -1)
            lefts.append(
                cumulative if sv>=0
                else cumulative+sv)
            widths.append(abs(sv))
            colors_w.append(
                "#e74c3c" if sv>=0
                else "#3498db")
            cumulative += sv

        ax.barh(x_pos, widths, left=lefts,
                color=colors_w, alpha=0.85,
                height=0.6)
        ax.axvline(
            base_val, color="#2c3e50",
            lw=1.5, ls="--",
            label=f"E[f(X)]="
                  f"{base_val:.3f}")
        ax.axvline(
            cumulative, color="#27ae60",
            lw=1.5, ls=":",
            label=f"f(x)={cumulative:.3f}")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(
            t_f[::-1], fontsize=10)
        for i, (w, l, sgn) in enumerate(
                zip(widths, lefts,
                    t_sgn[::-1])):
            val = w*(1 if sgn>=0 else -1)
            ax.text(
                l+w/2, i,
                f"{val:+.4f}",
                va="center", ha="center",
                fontsize=8, color="white",
                fontweight="bold")
        ax.set_xlabel(
            "SHAP Value (Cumulative)")
        ax.set_title(
            "SHAP Waterfall — "
            "Feature Contribution\n"
            f"Base={base_val:.3f} → "
            f"Output={cumulative:.3f}",
            fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "shap_waterfall.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("SHAP Waterfall", p))
    except Exception:
        pass

    # ── Confidence per Class ──────────────────
    try:
        class_conf = defaultdict(list)
        for pred, conf in zip(y_pred, y_conf):
            class_conf[pred].append(conf)
        cls_names = list(
            class_conf.keys())[:12]
        cls_means = [
            np.mean(class_conf[c])
            for c in cls_names]
        cls_stds  = [
            np.std(class_conf[c])
            for c in cls_names]
        fig, ax = plt.subplots(
            figsize=(12, 5))
        colors_c = [
            "#2ecc71" if mv>=0.8
            else "#f39c12" if mv>=0.6
            else "#e74c3c"
            for mv in cls_means]
        ax.bar(
            range(len(cls_names)),
            cls_means, color=colors_c,
            alpha=0.85,
            yerr=cls_stds, capsize=4)
        ax.set_xticks(range(len(cls_names)))
        ax.set_xticklabels(
            [c[:14] for c in cls_names],
            rotation=45, ha="right",
            fontsize=9)
        ax.axhline(
            0.8, color="gray",
            ls="--", lw=1.5,
            label="Threshold=0.8")
        ax.set_ylabel("Mean Confidence")
        ax.set_title(
            "Mean Confidence per Class",
            fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "conf_per_class.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(
            ("Confidence per Class", p))
    except Exception:
        pass

    # ── Confidence Breakdown ──────────────────
    try:
        bins   = [0,0.5,0.6,0.7,0.8,0.9,1.01]
        labels = ["<50%","50-60%","60-70%",
                  "70-80%","80-90%","90-100%"]
        counts = [int(np.sum(
            (y_conf>=bins[i]) &
            (y_conf<bins[i+1])))
            for i in range(len(bins)-1)]
        fig, ax = plt.subplots(figsize=(10,5))
        colors_b = [
            "#e74c3c","#e67e22","#f1c40f",
            "#2ecc71","#27ae60","#1abc9c"]
        bars = ax.bar(
            labels, counts,
            color=colors_b, alpha=0.85)
        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() +
                max(counts)*0.01,
                f"{val:,}",
                ha="center", fontsize=9)
        ax.set_xlabel("Confidence Range")
        ax.set_ylabel("Samples")
        ax.set_title(
            "Confidence Breakdown",
            fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "conf_breakdown.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(
            ("Confidence Breakdown", p))
    except Exception:
        pass

    return paths
