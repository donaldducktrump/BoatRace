import os
import re
import glob
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_K_DIR = os.path.join(DATA_DIR, "raw_data", "k_data")
PROC_DIR = os.path.join(DATA_DIR, "processed_data")
TRIFECTA_DIR = os.path.join(PROC_DIR, "trifecta_odds")
ODDS_CSV = os.path.join(PROC_DIR, "odds_dataframe", "odds_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "calibration_plots_sklearn")

# Settings
TAKEOUT_RATE = 0.75  # 控除率（払戻率）
N_BINS = 10
BIN_STRATEGY = "quantile"


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


from matplotlib import font_manager as _fm


def _setup_japanese_font() -> str | None:
    candidates = [
        "Meiryo",
        "Yu Gothic",
        "Yu Gothic UI",
        "MS Gothic",
        "MS PGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
    ]
    for name in candidates:
        try:
            path = _fm.findfont(name, fallback_to_default=False)
            if path and os.path.exists(path):
                plt.rcParams['font.family'] = name
                plt.rcParams['axes.unicode_minus'] = False
                return name
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    return None


def build_race_id(date_str: str, jcd: str, race: int | str) -> str:
    d = re.sub(r"[^0-9]", "", date_str)
    if len(d) == 8:
        yyyymmdd = d
    else:
        parts = re.split(r"[^0-9]", date_str)
        parts = [p for p in parts if p]
        if len(parts) == 3:
            y, m, day = parts
            yyyymmdd = f"{int(y):04d}{int(m):02d}{int(day):02d}"
        else:
            raise ValueError(f"Unexpected date format: {date_str}")
    return f"{yyyymmdd}{int(jcd):02d}{int(race):02d}"


def load_trifecta_board() -> pd.DataFrame:
    all_files = sorted(glob.glob(os.path.join(TRIFECTA_DIR, "*", "trifecta_*.csv")))
    frames = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
            df = df.rename(columns={c: c.strip() for c in df.columns})
            for k in ["Date", "JCD", "Race", "Boat1", "Boat2", "Boat3", "Odds"]:
                if k not in df.columns:
                    raise ValueError
            df["RaceID"] = df.apply(lambda r: build_race_id(str(r["Date"]), str(r["JCD"]), r["Race"]), axis=1)
            frames.append(df[["RaceID", "Boat1", "Boat2", "Boat3", "Odds"]].copy())
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["RaceID", "Boat1", "Boat2", "Boat3", "Odds"])
    df = pd.concat(frames, ignore_index=True)
    for col in ["Boat1", "Boat2", "Boat3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["Odds"] = pd.to_numeric(df["Odds"], errors="coerce")
    return df.dropna(subset=["Boat1", "Boat2", "Boat3", "Odds"]).copy()


def parse_k_winners() -> pd.DataFrame:
    recs: List[Dict] = []
    k_files = sorted(glob.glob(os.path.join(RAW_K_DIR, "*", "K*.TXT")))
    for fp in k_files:
        try:
            with open(fp, encoding="shift-jis", errors="ignore") as f:
                lines = [ln.rstrip("\n").replace("\u3000", " ") for ln in f]
        except Exception:
            continue
        date = None
        place_code = None
        for ln in lines:
            if place_code is None and "KBGN" in ln:
                m = re.search(r"(\d{2})KBGN", ln)
                if m:
                    place_code = m.group(1)
            if date is None and re.search(r"\d{4}[/\-]\s*\d{1,2}[/\-]\s*\d{1,2}", ln):
                m = re.search(r"(\d{4})[/\-]\s*(\d{1,2})[/\-]\s*(\d{1,2})", ln)
                if m:
                    y, mth, d = m.groups()
                    date = f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
            if date and place_code:
                break
        if not (date and place_code):
            continue
        pat = re.compile(r"^\s*(\d{1,2})R\s+(\d)-(\d)-(\d)\s+\d+\s+(\d)-(\d)-(\d)\s+\d+\s+(\d)-(\d)\s+\d+\s+(\d)-(\d)\s+\d+")
        for ln in lines:
            m = pat.match(ln)
            if not m:
                continue
            race = int(m.group(1))
            b1, b2, b3 = int(m.group(2)), int(m.group(3)), int(m.group(4))
            race_id = f"{re.sub(r'[^0-9]', '', date)}{place_code}{race:02d}"
            recs.append({"RaceID": race_id, "First": b1, "Second": b2, "Third": b3})
    if not recs:
        return pd.DataFrame(columns=["RaceID", "First", "Second", "Third"])
    return pd.DataFrame.from_records(recs).drop_duplicates(subset=["RaceID"])  # one per race


def find_boat_column(df: pd.DataFrame) -> str | None:
    candidate_cols = []
    for c in df.columns:
        if c in ("win_odds", "place_odds", "win_odds_mean", "レースID"):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().all():
            continue
        s = set(vals.dropna().unique().tolist())
        if all((isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer())) for x in s):
            if s.issubset({1, 2, 3, 4, 5, 6}):
                candidate_cols.append(c)
    for c in candidate_cols:
        g = df.groupby("レースID")[c].nunique().dropna()
        if (g == 6).mean() > 0.8:
            return c
    return candidate_cols[0] if candidate_cols else None


def load_win_place_odds() -> pd.DataFrame:
    if not os.path.exists(ODDS_CSV):
        return pd.DataFrame(columns=["レースID", "艇番", "win_odds", "place_odds"])
    df = pd.read_csv(ODDS_CSV, encoding_errors="ignore")
    if "レースID" not in df.columns or "win_odds" not in df.columns:
        raise RuntimeError("odds_data.csv must include レースID, win_odds")
    boat_col = find_boat_column(df)
    if boat_col is None:
        raise RuntimeError("Could not detect boat number column in odds_data.csv")
    sub = df[["レースID", boat_col, "win_odds"]].copy()
    sub = sub.rename(columns={boat_col: "Boat"})
    sub["Boat"] = pd.to_numeric(sub["Boat"], errors="coerce").astype("Int64")
    sub["win_odds"] = pd.to_numeric(sub["win_odds"], errors="coerce")
    return sub.dropna(subset=["Boat"]).copy()


def prob_from_odds_takeout(df: pd.DataFrame, odds_col: str, out_col: str, takeout: float) -> pd.DataFrame:
    d = df.copy()
    d[odds_col] = pd.to_numeric(d[odds_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[odds_col])
    d = d[d[odds_col] > 0].copy()
    d[out_col] = (takeout / d[odds_col]).clip(lower=0, upper=1)
    return d


def compute_marginals_from_trifecta_takeout(tri: pd.DataFrame, takeout: float) -> pd.DataFrame:
    if tri.empty:
        return pd.DataFrame(columns=["RaceID", "Boat", "p1", "p2", "p3"])
    t = tri.copy()
    t["Odds"] = pd.to_numeric(t["Odds"], errors="coerce")
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["Odds"]).copy()
    t = t[t["Odds"] > 0]
    t["p_trifecta"] = (takeout / t["Odds"]).clip(lower=0, upper=1)
    p1 = t.groupby(["RaceID", "Boat1"])['p_trifecta'].sum().rename("p1").reset_index().rename(columns={"Boat1": "Boat"})
    p2 = t.groupby(["RaceID", "Boat2"])['p_trifecta'].sum().rename("p2").reset_index().rename(columns={"Boat2": "Boat"})
    p3 = t.groupby(["RaceID", "Boat3"])['p_trifecta'].sum().rename("p3").reset_index().rename(columns={"Boat3": "Boat"})
    marg = pd.merge(p1, p2, on=["RaceID", "Boat"], how="outer")
    marg = pd.merge(marg, p3, on=["RaceID", "Boat"], how="outer")
    return marg.fillna(0.0)


def plot_calibration_sklearn(pred: np.ndarray, truth: np.ndarray, title: str, out_path: str, n_bins: int = N_BINS, strategy: str = BIN_STRATEGY) -> None:
    eps = 1e-6
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=int)
    m = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[m]
    truth = truth[m]
    if pred.size == 0:
        return
    pred = np.clip(pred, eps, 1 - eps)
    bs = brier_score_loss(truth, pred)
    prob_true, prob_pred = calibration_curve(truth, pred, n_bins=n_bins, strategy=strategy)
    prob_true = np.clip(prob_true, eps, 1.0)
    prob_pred = np.clip(prob_pred, eps, 1.0)

    plt.figure(figsize=(5, 5))
    grid = np.logspace(np.log10(eps), 0, 200)
    plt.plot(grid, grid, linestyle='--', color='gray', label='Ideal')
    plt.plot(prob_pred, prob_true, marker='o', color='C0', label=f"Calib (Brier={bs:.4f}, N={pred.size})")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('予測確率 (log)')
    plt.ylabel('実測確率 (log)')
    plt.title(title)
    # plt.xlim((eps, 1))
    # plt.ylim((eps, 1))
    plt.grid(alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    font_name = _setup_japanese_font()
    ensure_outdir(OUT_DIR)
    print(f"[setup] font={font_name or 'default'} out={OUT_DIR}")

    # Truth per race
    print("[load] winners from K files ...")
    winners = parse_k_winners()
    truths = []
    for _, r in winners.iterrows():
        rid = str(r["RaceID"])
        f, s, t = int(r["First"]), int(r["Second"]), int(r["Third"]) 
        for b in range(1, 7):
            truths.append({"RaceID": rid, "Boat": b, "is1": int(b == f), "is2": int(b == s), "is3": int(b == t)})
    truth_df = pd.DataFrame(truths)
    if truth_df.empty:
        print("[warn] No winners parsed.")
        return

    # Boards and odds
    print("[load] trifecta board ...")
    tri = load_trifecta_board()
    print(f"[info] trifecta rows: {len(tri)}")

    print("[load] win/place odds ...")
    wp = load_win_place_odds()
    print(f"[info] win odds rows: {len(wp)}")

    # Probabilities
    tri_marg = compute_marginals_from_trifecta_takeout(tri, TAKEOUT_RATE)
    tri_marg["RaceID"] = tri_marg["RaceID"].astype(str)

    win_probs = prob_from_odds_takeout(wp.rename(columns={"レースID": "RaceID"}), "win_odds", "p1_win", TAKEOUT_RATE)[["RaceID", "Boat", "p1_win"]]
    win_probs["RaceID"] = win_probs["RaceID"].astype(str)

    base = truth_df.copy()

    # 単勝: 1着
    df = base.merge(win_probs, on=["RaceID", "Boat"], how="inner")
    if not df.empty:
        out = os.path.join(OUT_DIR, "tansho_1st_sklearn.png")
        print(f"[plot] tansho_1st -> {out}")
        plot_calibration_sklearn(df["p1_win"].values, df["is1"].values, title="単勝: 1着 (scikit-learn)", out_path=out)

    # 二連単: 1,2着（三連単の周辺化）
    df = base.merge(tri_marg, on=["RaceID", "Boat"], how="inner")
    if not df.empty:
        out = os.path.join(OUT_DIR, "niren_tan_1st_sklearn.png")
        print(f"[plot] niren_tan_1st -> {out}")
        plot_calibration_sklearn(df["p1"].values, df["is1"].values, title="二連単: 1着 (scikit-learn)", out_path=out)
        out = os.path.join(OUT_DIR, "niren_tan_2nd_sklearn.png")
        print(f"[plot] niren_tan_2nd -> {out}")
        plot_calibration_sklearn(df["p2"].values, df["is2"].values, title="二連単: 2着 (scikit-leーン)", out_path=out)

    # 三連単: 1,2,3着
    df = base.merge(tri_marg, on=["RaceID", "Boat"], how="inner")
    if not df.empty:
        out = os.path.join(OUT_DIR, "sanren_tan_1st_sklearn.png")
        print(f"[plot] sanren_tan_1st -> {out}")
        plot_calibration_sklearn(df["p1"].values, df["is1"].values, title="三連単: 1着 (scikit-learn)", out_path=out)
        out = os.path.join(OUT_DIR, "sanren_tan_2nd_sklearn.png")
        print(f"[plot] sanren_tan_2nd -> {out}")
        plot_calibration_sklearn(df["p2"].values, df["is2"].values, title="三連単: 2着 (scikit-learn)", out_path=out)
        out = os.path.join(OUT_DIR, "sanren_tan_3rd_sklearn.png")
        print(f"[plot] sanren_tan_3rd -> {out}")
        plot_calibration_sklearn(df["p3"].values, df["is3"].values, title="三連単: 3着 (scikit-learn)", out_path=out)

    print(f"[done] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

