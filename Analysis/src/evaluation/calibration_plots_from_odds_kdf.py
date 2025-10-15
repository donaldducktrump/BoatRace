import os
import re
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_K_DIR = os.path.join(DATA_DIR, "raw_data", "k_data")
PROC_DIR = os.path.join(DATA_DIR, "processed_data")
K_DF_DIR = os.path.join(PROC_DIR, "k_dataframe")
TRIFECTA_DIR = os.path.join(PROC_DIR, "trifecta_odds")
TRIO_DIR = os.path.join(PROC_DIR, "trio_odds")
ODDS_CSV = os.path.join(PROC_DIR, "odds_dataframe", "odds_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "calibration_plots_df")

# Settings
TAKEOUT_RATE = 0.75  # 控除率
ONLY_TANSHO_KEI = True  # 単勝・二連単・三連単のみ


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
            # RaceID
            def build_race_id(date_str: str, jcd: str, race: int | str) -> str:
                d = re.sub(r"[^0-9]", "", str(date_str))
                if len(d) != 8:
                    parts = re.split(r"[^0-9]", str(date_str))
                    parts = [p for p in parts if p]
                    if len(parts) == 3:
                        y, m, day = parts
                        d = f"{int(y):04d}{int(m):02d}{int(day):02d}"
                return f"{d}{int(jcd):02d}{int(race):02d}"

            df["RaceID"] = df.apply(lambda r: build_race_id(r["Date"], r["JCD"], r["Race"]), axis=1)
            frames.append(df[["RaceID", "Boat1", "Boat2", "Boat3", "Odds"]].copy())
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["RaceID", "Boat1", "Boat2", "Boat3", "Odds"])
    out = pd.concat(frames, ignore_index=True)
    for col in ["Boat1", "Boat2", "Boat3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["Odds"] = pd.to_numeric(out["Odds"], errors="coerce")
    return out.dropna(subset=["Boat1", "Boat2", "Boat3", "Odds"]).copy()


def load_win_place_odds() -> pd.DataFrame:
    if not os.path.exists(ODDS_CSV):
        return pd.DataFrame(columns=["レースID", "艇番", "win_odds", "place_odds"])
    df = pd.read_csv(ODDS_CSV, encoding_errors="ignore")
    if "レースID" not in df.columns or "win_odds" not in df.columns:
        raise RuntimeError("odds_data.csv must include レースID, win_odds")
    # detect boat column in odds CSV
    def find_boat_column(dfo: pd.DataFrame) -> str | None:
        cands = []
        for c in dfo.columns:
            if c in ("win_odds", "place_odds", "win_odds_mean", "レースID"):
                continue
            vals = pd.to_numeric(dfo[c], errors="coerce")
            if vals.isna().all():
                continue
            s = set(vals.dropna().unique().tolist())
            if all((isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer())) for x in s):
                if s.issubset({1, 2, 3, 4, 5, 6}):
                    cands.append(c)
        for c in cands:
            g = dfo.groupby("レースID")[c].nunique().dropna()
            if (g == 6).mean() > 0.8:
                return c
        return cands[0] if cands else None

    boat_col = find_boat_column(df)
    if boat_col is None:
        raise RuntimeError("Could not detect boat column in odds_data.csv")
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


def trifecta_probs_takeout(tri: pd.DataFrame, takeout: float) -> pd.DataFrame:
    t = tri.copy()
    t["Odds"] = pd.to_numeric(t["Odds"], errors="coerce")
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["Odds"]).copy()
    t = t[t["Odds"] > 0]
    t["p_trifecta"] = (takeout / t["Odds"]).clip(lower=0, upper=1)
    return t


def compute_marginals_from_trifecta_takeout(tri: pd.DataFrame, takeout: float) -> pd.DataFrame:
    if tri.empty:
        return pd.DataFrame(columns=["RaceID", "Boat", "p1", "p2", "p3"])
    t = trifecta_probs_takeout(tri, takeout)
    p1 = t.groupby(["RaceID", "Boat1"])['p_trifecta'].sum().rename("p1").reset_index().rename(columns={"Boat1": "Boat"})
    p2 = t.groupby(["RaceID", "Boat2"])['p_trifecta'].sum().rename("p2").reset_index().rename(columns={"Boat2": "Boat"})
    p3 = t.groupby(["RaceID", "Boat3"])['p_trifecta'].sum().rename("p3").reset_index().rename(columns={"Boat3": "Boat"})
    marg = pd.merge(p1, p2, on=["RaceID", "Boat"], how="outer")
    marg = pd.merge(marg, p3, on=["RaceID", "Boat"], how="outer")
    return marg.fillna(0.0)


_time_pat1 = re.compile(r"^(\d+)m(\d{1,2})s(\d)$")  # e.g., 1m49s9
_time_pat2 = re.compile(r"^(\d)\.(\d{2})\.(\d)$")  # e.g., 1.49.7


def _parse_time_to_seconds(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = _time_pat1.match(s)
    if m:
        mm, ss, d = m.groups()
        return int(mm) * 60 + int(ss) + int(d) * 0.1
    m = _time_pat2.match(s)
    if m:
        m1, s2, d = m.groups()
        return int(m1) * 60 + int(s2) + int(d) * 0.1
    return None


def parse_winners_from_k_dataframe() -> pd.DataFrame:
    """processed_data/k_dataframe から各レースの1-2-3着の艇番を抽出する。"""
    files = sorted(glob.glob(os.path.join(K_DF_DIR, "*", "K*.pkl")))
    recs: List[Dict] = []
    if not files:
        return pd.DataFrame(columns=["RaceID", "First", "Second", "Third"])

    # Load odds once for choosing boat-col if needed
    try:
        odds_df = load_win_place_odds().rename(columns={"レースID": "RaceID"})
        odds_df["RaceID"] = odds_df["RaceID"].astype(str)
    except Exception:
        odds_df = pd.DataFrame(columns=["RaceID", "Boat", "win_odds"])

    for fp in files:
        try:
            df = pd.read_pickle(fp)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        # detect race id column
        race_col = None
        for c in df.columns:
            if isinstance(c, str) and ("ID" in c or c == "レースID"):
                race_col = c
                break
        if race_col is None:
            race_col = df.columns[0]

        # detect time column
        time_col = None
        for c in df.columns:
            if df[c].dtype == object:
                vals = df[c].dropna().astype(str).head(200)
                if len(vals) == 0:
                    continue
                ok = 0
                for v in vals:
                    if _parse_time_to_seconds(v) is not None:
                        ok += 1
                if ok / max(len(vals), 1) >= 0.6:
                    time_col = c
                    break
        if time_col is None:
            # cannot proceed on this file
            continue

        # numeric candidate columns with values 1..6 mostly
        cand_cols: List[str] = []
        for c in df.columns:
            if c == race_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().all():
                continue
            vals = s.dropna()
            if vals.empty:
                continue
            sset = set(vals.unique().tolist())
            if sset.issubset({1, 2, 3, 4, 5, 6}):
                # check per-race uniqueness
                gr = pd.concat({"v": s, "rid": df[race_col]}, axis=1).dropna()
                nun = gr.groupby("rid")["v"].nunique()
                if not nun.empty and (nun == 6).mean() > 0.4:
                    cand_cols.append(c)

        if not cand_cols:
            continue

        # identify finish column by agreement with time-rank
        best_col = None
        best_match = -1.0
        for c in cand_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            tmp = df[[race_col, time_col]].copy()
            tmp["sec"] = tmp[time_col].apply(_parse_time_to_seconds)
            tmp = pd.concat([tmp, s.rename("cand")], axis=1).dropna()
            if tmp.empty:
                continue
            # rank within race
            tmp["rank"] = tmp.groupby(race_col)["sec"].rank(method="first")
            agree = (np.isclose(tmp["cand"], tmp["rank"]))
            score = float(agree.mean())
            if score > best_match:
                best_match = score
                best_col = c

        finish_col = best_col
        if finish_col is None:
            continue

        # choose boat column among remaining candidates
        remaining = [c for c in cand_cols if c != finish_col]
        if not remaining:
            continue
        # prefer the one that joins odds best
        boat_col = remaining[0]
        if not odds_df.empty:
            max_hits = -1
            for c in remaining:
                sub = df[[race_col, c]].copy()
                sub = sub.rename(columns={race_col: "RaceID", c: "Boat"})
                sub["RaceID"] = sub["RaceID"].astype(str)
                sub["Boat"] = pd.to_numeric(sub["Boat"], errors="coerce")
                m = sub.merge(odds_df[["RaceID", "Boat"]].drop_duplicates(), on=["RaceID", "Boat"], how="inner")
                hits = len(m)
                if hits > max_hits:
                    max_hits = hits
                    boat_col = c

        # now build winners per race using time ascending
        df2 = df[[race_col, time_col, boat_col]].copy()
        df2["sec"] = df2[time_col].apply(_parse_time_to_seconds)
        df2 = df2.dropna(subset=["sec"]).copy()
        for rid, grp in df2.groupby(race_col):
            g = grp.sort_values("sec", ascending=True)
            boats = pd.to_numeric(g[boat_col], errors="coerce").dropna().astype(int).tolist()
            if len(boats) >= 3:
                recs.append({"RaceID": str(rid), "First": boats[0], "Second": boats[1], "Third": boats[2]})

    if not recs:
        return pd.DataFrame(columns=["RaceID", "First", "Second", "Third"])
    winners = pd.DataFrame.from_records(recs).drop_duplicates(subset=["RaceID"])  # keep first per race
    return winners


def make_calibration_plot(pred: np.ndarray, truth: np.ndarray, title: str, out_path: str, n_bins: int = 10) -> None:
    pred = np.asarray(pred)
    truth = np.asarray(truth).astype(int)
    m = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[m]
    truth = truth[m]
    if pred.size == 0:
        return
    # quantile bins
    try:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(pred, qs))
        if edges.size < 3:
            edges = np.linspace(0, 1, n_bins + 1)
    except Exception:
        edges = np.linspace(0, 1, n_bins + 1)

    bin_idx = np.digitize(pred, edges, right=True)
    xs, ys, ns = [], [], []
    for b in range(1, len(edges) + 1):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        xs.append(pred[sel].mean())
        ys.append(truth[sel].mean())
        ns.append(int(sel.sum()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.scatter(xs, ys, c='C0')
    for x, y, n in zip(xs, ys, ns):
        plt.text(x, y, str(n), fontsize=8, ha='left', va='bottom')
    plt.xlabel('予測確率')
    plt.ylabel('実測確率')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    font_name = _setup_japanese_font()
    print(f"[setup] フォント: {font_name or '既定'}", flush=True)
    ensure_outdir(OUT_DIR)
    print(f"[setup] 出力先: {OUT_DIR}", flush=True)

    # Winners from processed k_dataframe
    print("[load] k_dataframe から着順抽出...", flush=True)
    winners = parse_winners_from_k_dataframe()
    if winners.empty:
        print("[warn] k_dataframe から着順を抽出できませんでした。", flush=True)
        return
    winners["RaceID"] = winners["RaceID"].astype(str)
    print(f"[load] winners: {len(winners)} レース", flush=True)

    # Truth long format per boat
    truths = []
    for _, r in winners.iterrows():
        rid = r["RaceID"]
        f, s, t = int(r["First"]), int(r["Second"]), int(r["Third"]) 
        for b in range(1, 7):
            truths.append({"RaceID": rid, "Boat": b, "is1": int(b == f), "is2": int(b == s), "is3": int(b == t)})
    truth_df = pd.DataFrame(truths)

    # Load boards/odds
    tri_ordered = load_trifecta_board()
    print(f"[load] 三連単行数: {len(tri_ordered)}", flush=True)
    wp = load_win_place_odds().rename(columns={"レースID": "RaceID"})
    wp["RaceID"] = wp["RaceID"].astype(str)
    print(f"[load] 単勝行数: {len(wp)}", flush=True)

    # Probabilities
    p_marg_tri = compute_marginals_from_trifecta_takeout(tri_ordered, TAKEOUT_RATE)
    if not p_marg_tri.empty:
        p_marg_tri["RaceID"] = p_marg_tri["RaceID"].astype(str)
    print(f"[calc] 三連単マージン: {len(p_marg_tri)}", flush=True)

    win_probs = prob_from_odds_takeout(wp.dropna(subset=["win_odds"]).copy(), "win_odds", "p1_win", TAKEOUT_RATE)[["RaceID", "Boat", "p1_win"]]
    if not win_probs.empty:
        win_probs["RaceID"] = win_probs["RaceID"].astype(str)
    print(f"[calc] 単勝確率: {len(win_probs)}", flush=True)

    base = truth_df.copy()

    # 単勝
    if not win_probs.empty:
        df = base.merge(win_probs, on=["RaceID", "Boat"], how="inner")
        if not df.empty:
            print("[plot] 単勝: 1着 -> tansho_1st.png", flush=True)
            make_calibration_plot(df["p1_win"].values, df["is1"].values, title="単勝: 1着", out_path=os.path.join(OUT_DIR, "tansho_1st.png"))

    # 二連単（三連単由来）
    if not ONLY_TANSHO_KEI:
        pass
    # 単勝系: 二連単/三連単も出力
    if not p_marg_tri.empty:
        df = base.merge(p_marg_tri, on=["RaceID", "Boat"], how="inner")
        if not df.empty:
            print("[plot] 二連単: 1着 -> niren_tan_1st.png", flush=True)
            make_calibration_plot(df["p1"].clip(0,1).values, df["is1"].values, title="二連単: 1着 (三連単オッズ由来, r/odds)", out_path=os.path.join(OUT_DIR, "niren_tan_1st.png"))
            print("[plot] 二連単: 2着 -> niren_tan_2nd.png", flush=True)
            make_calibration_plot(df["p2"].clip(0,1).values, df["is2"].values, title="二連単: 2着 (三連単オッズ由来, r/odds)", out_path=os.path.join(OUT_DIR, "niren_tan_2nd.png"))

            print("[plot] 三連単: 1着 -> sanren_tan_1st.png", flush=True)
            make_calibration_plot(df["p1"].clip(0,1).values, df["is1"].values, title="三連単: 1着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_1st.png"))
            print("[plot] 三連単: 2着 -> sanren_tan_2nd.png", flush=True)
            make_calibration_plot(df["p2"].clip(0,1).values, df["is2"].values, title="三連単: 2着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_2nd.png"))
            print("[plot] 三連単: 3着 -> sanren_tan_3rd.png", flush=True)
            make_calibration_plot(df["p3"].clip(0,1).values, df["is3"].values, title="三連単: 3着 (r/odds)", out_path=os.path.join(OUT_DIR, "sanren_tan_3rd.png"))

    print(f"[done] 画像保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()

