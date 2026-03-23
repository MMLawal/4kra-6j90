#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ============================================================
# SETTINGS
# ============================================================

LIE_DIR = "lie_output"
RESIDENCE_CSV = "residence_results/contact_persistence_all.csv"
MSM_DIR = "msm_results"
STRUCT_DIR = "structural_stability_results"

OUTDIR = "ml_binding_results"
os.makedirs(OUTDIR, exist_ok=True)

WINDOW = 100
STEP = 50
RANDOM_STATE = 42

SYSTEM_KEY_MAP = {
    ("4kra", "Mangiferin"): "4kra_mangi",
    ("4kra", "Ciprofloxacin"): "4kra_cipro",
    ("4kra", "Azadirachtin"): "4kra_azad",
    ("dnagyrase", "Mangiferin"): "dnagyrase_mangi",
    ("dnagyrase", "Ciprofloxacin"): "dnagyrase_cipro",
    ("dnagyrase", "Azadirachtin"): "dnagyrase_azad",
}

# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def keep_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

def drop_all_nan_feature_columns(df, protected_cols):
    feature_cols = [c for c in df.columns if c not in protected_cols]
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        print("\nDropping all-NaN columns:\n", all_nan)
        df = df.drop(columns=all_nan)
    return df

# ============================================================
# LOADERS
# ============================================================

def load_lie_timeseries():
    rows = []

    for fp in sorted(glob.glob(os.path.join(LIE_DIR, "*_lie.dat"))):
        tag = os.path.basename(fp).replace("_lie.dat", "")

        try:
            data = np.loadtxt(fp)
        except Exception:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 3:
            continue

        vdw = data[:, 1]
        elec = data[:, 2]
        eint = vdw + elec

        for i in range(len(eint)):
            rows.append({
                "system_tag": tag,
                "frame_idx": i,
                "vdw": float(vdw[i]),
                "elec": float(elec[i]),
                "interaction_E": float(eint[i]),
            })

    return pd.DataFrame(rows)

def load_contact_persistence():
    if not os.path.exists(RESIDENCE_CSV):
        return pd.DataFrame()
    return pd.read_csv(RESIDENCE_CSV)

def load_structural_summary():
    fp = os.path.join(STRUCT_DIR, "structural_stability_summary.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)

def load_structural_timeseries():
    rows = []
    for fp in glob.glob(os.path.join(STRUCT_DIR, "*", "structural_metrics_timeseries.csv")):
        parent = os.path.basename(os.path.dirname(fp))
        df = pd.read_csv(fp)
        df["system_tag"] = parent
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_msm_features():
    """
    Loads msm_results/<system>/features.csv
    Example columns:
    frame,time_ns,ligand_rmsd,dist_res94,...,pocket_rg,contact_count
    """
    rows = []
    for fp in glob.glob(os.path.join(MSM_DIR, "*", "features.csv")):
        parent = os.path.basename(os.path.dirname(fp))   # e.g. dnagyrase_azad
        df = pd.read_csv(fp)
        df["system_tag"] = parent
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_msm_summary():
    fp = os.path.join(MSM_DIR, "msm_summary_all_systems.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)

def build_system_metadata():
    rows = []
    for (protein, ligand), tag in SYSTEM_KEY_MAP.items():
        rows.append({
            "protein": protein,
            "ligand": ligand,
            "system_tag": tag
        })
    return pd.DataFrame(rows)

# ============================================================
# WINDOW DATASET
# ============================================================

def make_window_features(lie_df, struct_ts_df, msm_feat_df):
    samples = []

    common_tags = set(lie_df["system_tag"].unique())

    if not struct_ts_df.empty:
        common_tags = common_tags & set(struct_ts_df["system_tag"].unique())
    if not msm_feat_df.empty:
        common_tags = common_tags & set(msm_feat_df["system_tag"].unique())

    common_tags = sorted(common_tags)

    for tag in common_tags:
        lie_sub = lie_df[lie_df["system_tag"] == tag].reset_index(drop=True)

        n = len(lie_sub)

        if not struct_ts_df.empty:
            st_sub = struct_ts_df[struct_ts_df["system_tag"] == tag].reset_index(drop=True)
            n = min(n, len(st_sub))
        else:
            st_sub = pd.DataFrame()

        if not msm_feat_df.empty:
            mf_sub = msm_feat_df[msm_feat_df["system_tag"] == tag].reset_index(drop=True)
            n = min(n, len(mf_sub))
        else:
            mf_sub = pd.DataFrame()

        if n < WINDOW:
            continue

        lie_sub = lie_sub.iloc[:n].copy()
        if not st_sub.empty:
            st_sub = st_sub.iloc[:n].copy()
        if not mf_sub.empty:
            mf_sub = mf_sub.iloc[:n].copy()

        # find msm distance columns
        msm_dist_cols = []
        if not mf_sub.empty:
            msm_dist_cols = [c for c in mf_sub.columns if c.startswith("dist_res")]

        for start in range(0, n - WINDOW + 1, STEP):
            end = start + WINDOW

            liew = lie_sub.iloc[start:end]
            feat = {
                "system_tag": tag,
                "window_start": start,
                "window_end": end,
                "Eint_mean": liew["interaction_E"].mean(),
                "Eint_std": liew["interaction_E"].std(),
                "Eint_min": liew["interaction_E"].min(),
                "Eint_max": liew["interaction_E"].max(),
                "vdw_mean": liew["vdw"].mean(),
                "elec_mean": liew["elec"].mean(),
            }

            # structural timeseries
            if not st_sub.empty:
                stw = st_sub.iloc[start:end]
                feat.update({
                    "lig_internal_rmsd_mean": stw["ligand_internal_rmsd_A"].mean() if "ligand_internal_rmsd_A" in stw.columns else np.nan,
                    "lig_internal_rmsd_std": stw["ligand_internal_rmsd_A"].std() if "ligand_internal_rmsd_A" in stw.columns else np.nan,
                    "pocket_rmsd_mean": stw["pocket_rmsd_A"].mean() if "pocket_rmsd_A" in stw.columns else np.nan,
                    "pocket_rmsd_std": stw["pocket_rmsd_A"].std() if "pocket_rmsd_A" in stw.columns else np.nan,
                    "lig_sasa_mean": stw["ligand_sasa_A2"].mean() if "ligand_sasa_A2" in stw.columns else np.nan,
                    "lig_sasa_std": stw["ligand_sasa_A2"].std() if "ligand_sasa_A2" in stw.columns else np.nan,
                    "pocket_sasa_mean": stw["pocket_sasa_A2"].mean() if "pocket_sasa_A2" in stw.columns else np.nan,
                    "pocket_sasa_std": stw["pocket_sasa_A2"].std() if "pocket_sasa_A2" in stw.columns else np.nan,
                })

            # msm features.csv window summaries
            if not mf_sub.empty:
                mfw = mf_sub.iloc[start:end]
                feat.update({
                    "msm_ligand_rmsd_mean": mfw["ligand_rmsd"].mean() if "ligand_rmsd" in mfw.columns else np.nan,
                    "msm_ligand_rmsd_std": mfw["ligand_rmsd"].std() if "ligand_rmsd" in mfw.columns else np.nan,
                    "msm_pocket_rg_mean": mfw["pocket_rg"].mean() if "pocket_rg" in mfw.columns else np.nan,
                    "msm_pocket_rg_std": mfw["pocket_rg"].std() if "pocket_rg" in mfw.columns else np.nan,
                    "msm_contact_count_mean": mfw["contact_count"].mean() if "contact_count" in mfw.columns else np.nan,
                    "msm_contact_count_std": mfw["contact_count"].std() if "contact_count" in mfw.columns else np.nan,
                })

                # summarize contact-distance pattern
                if msm_dist_cols:
                    feat["msm_dist_mean_all"] = mfw[msm_dist_cols].mean().mean()
                    feat["msm_dist_std_all"] = mfw[msm_dist_cols].stack().std()
                    feat["msm_close_contact_fraction"] = (mfw[msm_dist_cols] < 4.5).mean().mean()
                    feat["msm_very_close_contact_fraction"] = (mfw[msm_dist_cols] < 3.5).mean().mean()

            # target: window-level stability proxy
            lig_rmsd_term = feat.get("msm_ligand_rmsd_mean", feat.get("lig_internal_rmsd_mean", np.nan))
            pocket_term = feat.get("msm_pocket_rg_mean", feat.get("pocket_rmsd_mean", np.nan))
            sasa_term = feat.get("lig_sasa_mean", np.nan)
            contact_term = feat.get("msm_contact_count_mean", np.nan)

            feat["target_binding_stability"] = (
                -feat["Eint_mean"]
                - 2.0 * (0 if pd.isna(lig_rmsd_term) else lig_rmsd_term)
                - 0.01 * (0 if pd.isna(sasa_term) else sasa_term)
                - 1.0 * (0 if pd.isna(pocket_term) else pocket_term)
                + 0.5 * (0 if pd.isna(contact_term) else contact_term)
            )

            samples.append(feat)

    return pd.DataFrame(samples)

def add_system_level_features(window_df, meta_df, contact_df, struct_summary_df, msm_summary_df):
    out = window_df.merge(meta_df, on="system_tag", how="left")

    # contact persistence summaries
    if not contact_df.empty:
        csum = (
            contact_df.groupby(["protein", "ligand"])
            .agg(
                contact_persistence_mean=("persistence", "mean"),
                contact_persistence_max=("persistence", "max"),
                n_high_persist_contacts=("persistence", lambda x: np.sum(np.array(x) >= 0.9)),
            )
            .reset_index()
        )
        out = out.merge(csum, on=["protein", "ligand"], how="left")

    # structural summaries
    if not struct_summary_df.empty:
        ssum = struct_summary_df.copy()
        ssum["system_tag"] = ssum.apply(lambda r: SYSTEM_KEY_MAP.get((r["protein"], r["ligand"])), axis=1)
        keep_cols = keep_existing_cols(ssum, [
            "system_tag",
            "n_pocket_residues",
            "ligand_internal_rmsd_mean_A",
            "pocket_rmsd_mean_A",
            "ligand_sasa_mean_A2",
            "pocket_sasa_mean_A2",
            "schlitter_entropy_J_mol_K"
        ])
        out = out.merge(ssum[keep_cols], on="system_tag", how="left")

    # optional msm summary file
    if not msm_summary_df.empty:
        msum = msm_summary_df.copy()
        msum["system_tag"] = msum.apply(lambda r: SYSTEM_KEY_MAP.get((r["protein"], r["ligand"])), axis=1)
        keep_cols = keep_existing_cols(msum, [
            "system_tag",
            "bound_residence_time_proxy_ns",
            "bound_self_transition_prob",
            "bound_macro_ligand_rmsd_mean",
            "bound_macro_contact_count_mean",
            "bound_macro_pocket_rg_mean"
        ])
        out = out.merge(msum[keep_cols], on="system_tag", how="left")

    out["protein_is_4kra"] = (out["protein"] == "4kra").astype(int)
    return out

# ============================================================
# MODELING
# ============================================================

def evaluate_model(name, model, X, y, groups):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    n_groups = len(np.unique(groups))
    n_splits = min(3, n_groups)
    if n_splits < 2:
        raise RuntimeError("Need at least 2 systems/groups for grouped CV.")

    cv = GroupKFold(n_splits=n_splits)
    preds = cross_val_predict(pipe, X, y, groups=groups, cv=cv)

    metrics = {
        "model": name,
        "RMSE": rmse(y, preds),
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds)
    }
    return metrics, preds

def fit_feature_importance_rf(X, y):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    model.fit(X, y)
    rf = model.named_steps["model"]
    return rf.feature_importances_

# ============================================================
# MAIN
# ============================================================

def main():
    meta_df = build_system_metadata()
    lie_df = load_lie_timeseries()
    contact_df = load_contact_persistence()
    struct_summary_df = load_structural_summary()
    struct_ts_df = load_structural_timeseries()
    msm_feat_df = load_msm_features()
    msm_summary_df = load_msm_summary()

    if lie_df.empty:
        raise RuntimeError("No LIE timeseries found in lie_output.")
    if struct_ts_df.empty:
        print("Warning: no structural timeseries found.")
    if msm_feat_df.empty:
        print("Warning: no MSM features.csv files found.")

    window_df = make_window_features(lie_df, struct_ts_df, msm_feat_df)
    if window_df.empty:
        raise RuntimeError("No window-level samples were created.")

    data = add_system_level_features(
        window_df, meta_df, contact_df, struct_summary_df, msm_summary_df
    )

    protected = {"system_tag", "window_start", "window_end", "protein", "ligand", "target_binding_stability"}
    data = drop_all_nan_feature_columns(data, protected)

    data.to_csv(os.path.join(OUTDIR, "ml_dataset_windows.csv"), index=False)

    target = "target_binding_stability"
    feature_cols = [c for c in data.columns if c not in protected]

    # also remove non-numeric columns just in case
    numeric_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(data[c]):
            numeric_feature_cols.append(c)

    X = data[numeric_feature_cols]
    y = data[target].values
    groups = data["system_tag"].values

    results = []
    pred_table = data[["system_tag", "protein", "ligand", "window_start", "window_end", target]].copy()

    # RF
    rf_model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    m, preds = evaluate_model("RandomForest", rf_model, X, y, groups)
    results.append(m)
    pred_table["pred_rf"] = preds

    # XGB
    if HAS_XGB:
        xgb_model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4
        )
        m, preds = evaluate_model("XGBoost", xgb_model, X, y, groups)
        results.append(m)
        pred_table["pred_xgb"] = preds

    # NN
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    m, preds = evaluate_model("NeuralNet", nn_model, X, y, groups)
    results.append(m)
    pred_table["pred_nn"] = preds

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(OUTDIR, "model_metrics.csv"), index=False)
    pred_table.to_csv(os.path.join(OUTDIR, "window_predictions.csv"), index=False)

    # RF importance
    importances = fit_feature_importance_rf(X, y)
    fi_df = pd.DataFrame({
        "feature": numeric_feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(OUTDIR, "rf_feature_importance.csv"), index=False)

    # aggregate by system
    pred_cols = [c for c in pred_table.columns if c.startswith("pred_")]
    agg = pred_table.groupby(["protein", "ligand"])[pred_cols + [target]].mean().reset_index()
    agg.to_csv(os.path.join(OUTDIR, "system_level_predicted_stability.csv"), index=False)

    print("\nModel metrics\n")
    print(metrics_df.to_string(index=False))

    print("\nFeatures used\n")
    print(numeric_feature_cols)

    print("\nTop RF features\n")
    print(fi_df.head(15).to_string(index=False))

    print("\nPredicted system-level stability\n")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
