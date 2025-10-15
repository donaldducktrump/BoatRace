import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths (adjust if your structure changes)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_K_DIR = os.path.join(DATA_DIR, "raw_data", "k_data")
PROC_DIR = os.path.join(DATA_DIR, "processed_data")
TRIFECTA_DIR = os.path.join(PROC_DIR, "trifecta_odds")
TRIO_DIR = os.path.join(PROC_DIR, "trio_odds")
ODDS_CSV = os.path.join(PROC_DIR, "odds_dataframe", "odds_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "calibration_plots")

# Settings
TAKEOUT_RATE = 0.75  # 控除率（払戻率）
# 単勝系 = 単勝・二連単・三連単 のみを出力
ONLY_TANSHO_KEI = True

# Try to set a Japanese-capable font to avoid mojibake
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


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_race_id(date_str: str, jcd: str, race: int | str) -> str:
    # date_str: 'YYYY-MM-DD' or 'YYYY/MM/DD'
    d = re.sub(r"[^0-9]", "", date_str)  # keep digits only
    if len(d) == 8:
        yyyymmdd = d
    else:
        # try parse e.g., 2024-1-1
        parts = re.split(r"[^0-9]", date_str)
        parts = [p for p in parts if p]
        if len(parts) == 3:
            y, m, day = parts
            yyyymmdd = f"{int(y):04d}{int(m):02d}{int(day):02d}"
        else:
            raise ValueError(f"Unexpected date format: {date_str}")
    return f"{yyyymmdd}{int(jcd):02d}{int(race):02d}"


def load_trifecta_board() -> pd.DataFrame:
    # Columns: Date,JCD,Race,Boat1,Boat2,Boat3,Odds
    all_files = sorted(glob.glob(os.path.join(TRIFECTA_DIR, "*", "trifecta_*.csv")))
    frames = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
            # normalize column names expected
            exp = {"Date": "Date", "JCD": "JCD", "Race": "Race", "Boat1": "Boat1", "Boat2": "Boat2", "Boat3": "Boat3", "Odds": "Odds"}
            # tolerate casing variations
            df = df.rename(columns={c: c.strip() for c in df.columns})
            for k in list(exp.keys()):
                if k not in df.columns:
                    raise ValueError(f"Missing column {k} in {fp}")
            df = df[list(exp.keys())].copy()
            df["RaceID"] = df.apply(lambda r: build_race_id(str(r["Date"]), str(r["JCD"]), r["Race"]), axis=1)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["RaceID", "Boat1", "Boat2", "Boat3", "Odds"])  # empty
    df = pd.concat(frames, ignore_index=True)
    # Clean types
    for col in ["Boat1", "Boat2", "Boat3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["Odds"] = pd.to_numeric(df["Odds"], errors="coerce")
    df = df.dropna(subset=["Boat1", "Boat2", "Boat3", "Odds"]).copy()
    return df


def load_trio_board() -> pd.DataFrame:
    # Columns: Date,JCD,Race,Boat1,Boat2,Boat3,Odds (unordered triple)
    all_files = sorted(glob.glob(os.path.join(TRIO_DIR, "*", "trio_*.csv")))
    frames = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
            df = df.rename(columns={c: c.strip() for c in df.columns})
            for k in ["Date", "JCD", "Race", "Boat1", "Boat2", "Boat3", "Odds"]:
                if k not in df.columns:
                    raise ValueError(f"Missing column {k} in {fp}")
            df = df[["Date", "JCD", "Race", "Boat1", "Boat2", "Boat3", "Odds"]].copy()
            df["RaceID"] = df.apply(lambda r: build_race_id(str(r["Date"]), str(r["JCD"]), r["Race"]), axis=1)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["RaceID", "Boat1", "Boat2", "Boat3", "Odds"])  # empty
    df = pd.concat(frames, ignore_index=True)
    # Clean types
    for col in ["Boat1", "Boat2", "Boat3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["Odds"] = pd.to_numeric(df["Odds"], errors="coerce")
    df = df.dropna(subset=["Boat1", "Boat2", "Boat3", "Odds"]).copy()
    # Trio board should be order-insensitive; enforce sorted tuple to avoid duplicates
    df[["Boat1", "Boat2", "Boat3"]] = np.sort(df[["Boat1", "Boat2", "Boat3"]].values, axis=1)
    return df


def implied_prob_from_odds_group(df: pd.DataFrame, odds_col: str, prob_col: str) -> pd.DataFrame:
    # df must contain groups (by RaceID); convert odds to probs by inverse-odds normalization
    d = df.copy()
    d[odds_col] = pd.to_numeric(d[odds_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[odds_col])
    # remove zero or negative odds entries
    d = d[d[odds_col] > 0].copy()
    d["_w"] = 1.0 / d[odds_col]
    # normalize within RaceID
    denom = d.groupby("RaceID")["_w"].transform("sum")
    d[prob_col] = d["_w"] / denom
    d = d.drop(columns=["_w"]).copy()
    return d


def implied_prob_from_odds_group_with_target_sum(df: pd.DataFrame, odds_col: str, prob_col: str, target_sum: float) -> pd.DataFrame:
    # Use inverse odds weights and scale so that the sum within a race equals target_sum (e.g., 2 for place)
    d = df.copy()
    d[odds_col] = pd.to_numeric(d[odds_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[odds_col])
    d = d[d[odds_col] > 0].copy()
    d["_w"] = 1.0 / d[odds_col]
    sum_w = d.groupby("RaceID")["_w"].transform("sum")
    d[prob_col] = target_sum * d["_w"] / sum_w
    d = d.drop(columns=["_w"]).copy()
    return d


def prob_from_odds_takeout(df: pd.DataFrame, odds_col: str, out_col: str, takeout: float) -> pd.DataFrame:
    """Compute implied probability using takeout directly: p = takeout / odds.
    No normalization across a race. Assumes odds are payout per 1 unit.
    """
    d = df.copy()
    d[odds_col] = pd.to_numeric(d[odds_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[odds_col])
    d = d[d[odds_col] > 0].copy()
    d[out_col] = (takeout / d[odds_col]).clip(lower=0, upper=1)
    return d


def trifecta_probs_takeout(tri: pd.DataFrame, takeout: float) -> pd.DataFrame:
    """Add p_trifecta = takeout / Odds to trifecta board (no per-race normalization)."""
    t = tri.copy()
    t["Odds"] = pd.to_numeric(t["Odds"], errors="coerce")
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["Odds"]).copy()
    t = t[t["Odds"] > 0]
    t["p_trifecta"] = (takeout / t["Odds"]).clip(lower=0, upper=1)
    return t


def compute_marginals_from_trifecta_takeout(tri: pd.DataFrame, takeout: float) -> pd.DataFrame:
    """Compute per-boat P(1st), P(2nd), P(3rd) from 三連単 board using p = r/odds."""
    if tri.empty:
        return pd.DataFrame(columns=["RaceID", "Boat", "p1", "p2", "p3"])
    t = trifecta_probs_takeout(tri, takeout)
    p1 = t.groupby(["RaceID", "Boat1"])['p_trifecta'].sum().rename("p1").reset_index().rename(columns={"Boat1": "Boat"})
    p2 = t.groupby(["RaceID", "Boat2"])['p_trifecta'].sum().rename("p2").reset_index().rename(columns={"Boat2": "Boat"})
    p3 = t.groupby(["RaceID", "Boat3"])['p_trifecta'].sum().rename("p3").reset_index().rename(columns={"Boat3": "Boat"})
    marg = pd.merge(p1, p2, on=["RaceID", "Boat"], how="outer")
    marg = pd.merge(marg, p3, on=["RaceID", "Boat"], how="outer")
    return marg.fillna(0.0)


def parse_k_winners() -> pd.DataFrame:
    # Extract winners triple (first, second, third boat numbers) per RaceID from K files.
    # Relies on lines like: " 1R  1-3-6  1830  1-3-6   760  1-3   390  1-3   310"
    recs: List[Dict] = []
    k_files = sorted(glob.glob(os.path.join(RAW_K_DIR, "*", "K*.TXT")))
    for fp in k_files:
        try:
            with open(fp, encoding="shift-jis", errors="ignore") as f:
                lines = [ln.rstrip("\n").replace("\u3000", " ") for ln in f]
        except Exception:
            continue

        # find date and place code
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

        # regex to catch the payout summary lines with four columns (3連単,3連複,2連単,2連複)
        pat = re.compile(r"^\s*(\d{1,2})R\s+(\d)-(\d)-(\d)\s+\d+\s+(\d)-(\d)-(\d)\s+\d+\s+(\d)-(\d)\s+\d+\s+(\d)-(\d)\s+\d+")
        for ln in lines:
            m = pat.match(ln)
            if not m:
                continue
            race = int(m.group(1))
            b1, b2, b3 = int(m.group(2)), int(m.group(3)), int(m.group(4))  # 三連単の順序
            race_id = f"{re.sub(r'[^0-9]', '', date)}{place_code}{race:02d}"
            recs.append({
                "RaceID": race_id,
                "First": b1,
                "Second": b2,
                "Third": b3,
            })
    if not recs:
        return pd.DataFrame(columns=["RaceID", "First", "Second", "Third"])
    winners = pd.DataFrame.from_records(recs).drop_duplicates(subset=["RaceID"])  # keep first occurrence
    return winners


def find_boat_column(df: pd.DataFrame) -> str | None:
    # Try to detect the column that corresponds to boat number (1..6) per レースID.
    # Heuristic: numeric, values subset of {1,2,3,4,5,6} across dataset, and per RaceID we see 6 unique values.
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
    # Verify per race uniqueness
    for c in candidate_cols:
        g = df.groupby("レースID")[c].nunique().dropna()
        # majority of races should have 6 boats
        if (g == 6).mean() > 0.8:
            return c
    return candidate_cols[0] if candidate_cols else None


def load_win_place_odds() -> pd.DataFrame:
    if not os.path.exists(ODDS_CSV):
        return pd.DataFrame(columns=["レースID", "艇番", "win_odds", "place_odds"])
    df = pd.read_csv(ODDS_CSV, encoding_errors="ignore")
    if "レースID" not in df.columns or "win_odds" not in df.columns or "place_odds" not in df.columns:
        raise RuntimeError("odds_data.csv must include レースID, win_odds, place_odds")
    boat_col = find_boat_column(df)
    if boat_col is None:
        raise RuntimeError("Could not detect boat number column in odds_data.csv")
    sub = df[["レースID", boat_col, "win_odds", "place_odds"]].copy()
    sub = sub.rename(columns={boat_col: "Boat"})
    sub["Boat"] = pd.to_numeric(sub["Boat"], errors="coerce").astype("Int64")
    sub["win_odds"] = pd.to_numeric(sub["win_odds"], errors="coerce")
    sub["place_odds"] = pd.to_numeric(sub["place_odds"], errors="coerce")
    sub = sub.dropna(subset=["Boat"]).copy()
    return sub


def compute_marginals_from_trifecta(tri: pd.DataFrame) -> pd.DataFrame:
    # tri: columns RaceID, Boat1, Boat2, Boat3, Odds
    tri = implied_prob_from_odds_group(tri, "Odds", "p_trifecta")
    # 1st/2nd/3rd marginals per boat
    p1 = tri.groupby(["RaceID", "Boat1"])['p_trifecta'].sum().rename("p1").reset_index().rename(columns={"Boat1": "Boat"})
    p2 = tri.groupby(["RaceID", "Boat2"])['p_trifecta'].sum().rename("p2").reset_index().rename(columns={"Boat2": "Boat"})
    p3 = tri.groupby(["RaceID", "Boat3"])['p_trifecta'].sum().rename("p3").reset_index().rename(columns={"Boat3": "Boat"})
    marg = pd.merge(p1, p2, on=["RaceID", "Boat"], how="outer")
    marg = pd.merge(marg, p3, on=["RaceID", "Boat"], how="outer")
    marg = marg.fillna(0.0)
    # 2連単 marginals by summing over third boat
    # P(1=i,2=j) = sum_k p(i,j,k). Not required to store full matrix; we only need per-boat 1st and 2nd marginals which equal p1 and p2 already.
    return marg


def compute_top2_from_trifecta_for_quinella(tri: pd.DataFrame) -> pd.DataFrame:
    # Use trifecta probs to compute probability that a boat is in top2 (any order)
    tri = implied_prob_from_odds_group(tri, "Odds", "p_trifecta")
    # top2 for boat i: sum over all tuples where i is Boat1 or Boat2
    p_top2_1 = tri.groupby(["RaceID", "Boat1"])['p_trifecta'].sum().rename("a").reset_index().rename(columns={"Boat1": "Boat"})
    p_top2_2 = tri.groupby(["RaceID", "Boat2"])['p_trifecta'].sum().rename("b").reset_index().rename(columns={"Boat2": "Boat"})
    p_top2 = pd.merge(p_top2_1, p_top2_2, on=["RaceID", "Boat"], how="outer").fillna(0.0)
    p_top2["p_top2"] = p_top2["a"] + p_top2["b"]
    return p_top2[["RaceID", "Boat", "p_top2"]]


def compute_top3_from_trio(trio: pd.DataFrame) -> pd.DataFrame:
    # trio: unordered triple board with Odds
    trio = implied_prob_from_odds_group(trio, "Odds", "p_trio")
    # top3 for boat i: sum of probabilities of all triples containing i
    p_a = trio.groupby(["RaceID", "Boat1"])['p_trio'].sum().rename("x").reset_index().rename(columns={"Boat1": "Boat"})
    p_b = trio.groupby(["RaceID", "Boat2"])['p_trio'].sum().rename("y").reset_index().rename(columns={"Boat2": "Boat"})
    p_c = trio.groupby(["RaceID", "Boat3"])['p_trio'].sum().rename("z").reset_index().rename(columns={"Boat3": "Boat"})
    p_top3 = pd.merge(pd.merge(p_a, p_b, on=["RaceID", "Boat"], how="outer"), p_c, on=["RaceID", "Boat"], how="outer").fillna(0.0)
    p_top3["p_top3"] = p_top3["x"] + p_top3["y"] + p_top3["z"]
    return p_top3[["RaceID", "Boat", "p_top3"]]


def compute_top3_from_trio_for_wide(trio: pd.DataFrame) -> pd.DataFrame:
    # Identity: sum_{j!=i} P(i & j in top3) = 2 * P(i in top3)
    # Using trio board, P(i & j in top3) = sum_{k != i,j} P_trio({i,j,k}).
    trio = implied_prob_from_odds_group(trio, "Odds", "p_trio")
    recs = []
    for (rid, grp) in trio.groupby("RaceID"):
        # Build dict for quick lookup by unordered triple
        # rows already with sorted (Boat1<=Boat2<=Boat3)
        triples = grp[["Boat1", "Boat2", "Boat3", "p_trio"]].values
        # Enumerate pair mass
        pair_mass = defaultdict(float)  # key (i,j) with i<j
        for b1, b2, b3, p in triples:
            pair_mass[tuple(sorted((int(b1), int(b2))))] += float(p)
            pair_mass[tuple(sorted((int(b1), int(b3))))] += float(p)
            pair_mass[tuple(sorted((int(b2), int(b3))))] += float(p)
        # P_top3(i) = 0.5 * sum_{j!=i} pair_mass[min(i,j), max(i,j)]
        boats = set([int(x) for x in grp[["Boat1", "Boat2", "Boat3"]].values.reshape(-1).tolist()])
        for i in sorted(boats):
            s = 0.0
            for j in sorted(boats):
                if i == j:
                    continue
                k = (min(i, j), max(i, j))
                s += pair_mass.get(k, 0.0)
            p_top3 = 0.5 * s
            recs.append({"RaceID": rid, "Boat": i, "p_top3": p_top3})
    if not recs:
        return pd.DataFrame(columns=["RaceID", "Boat", "p_top3"])
    out = pd.DataFrame.from_records(recs)
    return out


def make_calibration_plot(pred: np.ndarray, truth: np.ndarray, title: str, out_path: str, n_bins: int = 10) -> None:
    # Bin by predicted probability (quantile bins for balanced counts)
    pred = np.asarray(pred)
    truth = np.asarray(truth).astype(int)
    # remove NaNs
    m = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[m]
    truth = truth[m]
    if pred.size == 0:
        return
    # quantile edges
    try:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(pred, qs))
        if edges.size < 3:
            # fallback to uniform bins
            edges = np.linspace(0, 1, n_bins + 1)
    except Exception:
        edges = np.linspace(0, 1, n_bins + 1)

    bin_idx = np.digitize(pred, edges, right=True)
    # bins are 1..len(edges)
    xs, ys, ns = [], [], []
    for b in range(1, len(edges) + 1):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        xs.append(pred[sel].mean())
        ys.append(truth[sel].mean())
        ns.append(sel.sum())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.scatter(xs, ys, c='C0')
    for x, y, n in zip(xs, ys, ns):
        plt.text(x, y, str(n), fontsize=8, ha='left', va='bottom')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical probability')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    _setup_japanese_font()
    ensure_outdir(OUT_DIR)

    # Load winners (truth)
    winners = parse_k_winners()
    if winners.empty:
        print("No winners parsed from K files. Check raw_data/k_data.")
    # Prepare truth long format per boat
    truths = []
    for _, r in winners.iterrows():
        rid = r["RaceID"]
        f, s, t = int(r["First"]), int(r["Second"]), int(r["Third"]) 
        for b in range(1, 7):
            truths.append({
                "RaceID": rid,
                "Boat": b,
                "is1": int(b == f),
                "is2": int(b == s),
                "is3": int(b == t),
            })
    truth_df = pd.DataFrame(truths)
    if not truth_df.empty:
        truth_df["RaceID"] = truth_df["RaceID"].astype(str)

    # Load boards
    tri_ordered = load_trifecta_board()
    trio_unordered = load_trio_board()

    # Load win/place
    wp = load_win_place_odds()

    # Compute probabilities per bet type
    # 三連単 marginals（控除率 r/odds 使用）
    p_marg_tri = compute_marginals_from_trifecta_takeout(tri_ordered, TAKEOUT_RATE) if not tri_ordered.empty else pd.DataFrame(columns=["RaceID", "Boat", "p1", "p2", "p3"])
    if not p_marg_tri.empty:
        p_marg_tri["RaceID"] = p_marg_tri["RaceID"].astype(str)
    # 三連複/拡連複は単勝系では使わない
    p_top3_trio = pd.DataFrame(columns=["RaceID", "Boat", "p_top3"]) 
    p_top3_wide = pd.DataFrame(columns=["RaceID", "Boat", "p_top3"]) 

    # 単勝（控除率を使用: p = r / odds）
    if not wp.empty:
        wp2 = wp.rename(columns={"レースID": "RaceID"})
        win_probs = prob_from_odds_takeout(wp2.dropna(subset=["win_odds"]).copy(), "win_odds", "p1_win", TAKEOUT_RATE)[
            ["RaceID", "Boat", "p1_win"]
        ]
        if not win_probs.empty:
            win_probs["RaceID"] = win_probs["RaceID"].astype(str)
        place_probs = pd.DataFrame(columns=["RaceID", "Boat", "p_top2_place"])  # 複勝は出力しない
    else:
        win_probs = pd.DataFrame(columns=["RaceID", "Boat", "p1_win"]) 
        place_probs = pd.DataFrame(columns=["RaceID", "Boat", "p_top2_place"]) 

    # Join with truth
    base = truth_df.copy()

    # Plot 単勝 (1着)
    if not win_probs.empty:
        df = base.merge(win_probs, on=["RaceID", "Boat"], how="inner")
        if not df.empty:
            make_calibration_plot(df["p1_win"].values, df["is1"].values, title="単勝: 1着", out_path=os.path.join(OUT_DIR, "tansho_1st.png"))

    # 複勝はスキップ

    # 単勝系（単勝・二連単・三連単）を出力
    # 二連単 (1,2着) – 三連単 board から p = r/odds を用いて周辺化
    if not p_marg_tri.empty:
        df = base.merge(p_marg_tri, on=["RaceID", "Boat"], how="inner")
        if not df.empty:
            make_calibration_plot(df["p1"].clip(0,1).values, df["is1"].values, title="二連単: 1着 (三連単オッズ由来, r/odds)", out_path=os.path.join(OUT_DIR, "niren_tan_1st.png"))
            make_calibration_plot(df["p2"].clip(0,1).values, df["is2"].values, title="二連単: 2着 (三連単オッズ由来, r/odds)", out_path=os.path.join(OUT_DIR, "niren_tan_2nd.png"))

    # 三連単 (1,2,3着) – 同様に r/odds 周辺化
    if not p_marg_tri.empty:
        df = base.merge(p_marg_tri, on=["RaceID", "Boat"], how="inner")
        if not df.empty:
            make_calibration_plot(df["p1"].clip(0,1).values, df["is1"].values, title="三連単: 1着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_1st.png"))
            make_calibration_plot(df["p2"].clip(0,1).values, df["is2"].values, title="三連単: 2着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_2nd.png"))
            make_calibration_plot(df["p3"].clip(0,1).values, df["is3"].values, title="三連単: 3着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_3rd.png"))

    print(f"Saved calibration plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
