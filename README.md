# BOATRACE 分析リポジトリ

## To-Do（今後やること）

- [ ] calibration plotの年次変化を見る
- [ ] （必要に応じて項目を追加してください）

補足: このチェックリストは GitHub 上でクリックして状態を更新できます（`[ ]` → `[x]`）。完了後は行を削除しても構いません。

---

## ディレクトリ構成（概要）

- `Analysis/`
  - `config/`: 実行環境・ログなどの設定
  - `data/`: データ管理
    - `raw_data/`: 生データ（例: `before_data*`, `b_data`, `k_data`, `trifecta_data` など）
    - `processed_data/`: 前処理後データ（例: `odds_dataframe`, `trifecta_odds`, `trio_odds` など）
  - `docs/`: ドキュメント（API、レポート、仕様、参考論文など）
  - `models/`: 学習済みモデル・チェックポイント・エクスポート
  - `notebooks/`: 分析ノートブック
  - `results/`: 生成物
    - `calibration_plots/`: キャリブレーションプロット出力（例: `Analysis/results/calibration_plots/tansho_1st.png`）
    - `calibration_plots_sklearn/`
    - `feature_importance/`
  - `src/`:
    - `data_processing/`: 取得・前処理（例: `get_before_info/`, `get_race_data/`）
    - `evaluation/`: 評価スクリプト（例: `calibration_plots_from_odds.py`, `calibration_plots_sklearn.py`, `inspection*.py`）
    - `model_training/`: 学習（例: `dnn.py`, `with_before_data/`, `with_fan/`）
    - `utils/`, `tests/`
- `ML_pred/`: 旧来の実験スクリプトや中間生成物（`before_data*`, `before_dataframe*`, `odds_dataframe` などを含む）
- `Poseidon/`: データ収集・集計関連（スクレイピングや集計用スクリプト、`*.sqlite`/`*.json` などの生成物）
- `Papers/`: 参考論文類
- `boatrace/`: Python 仮想環境（venv）。通常はリポジトリに含めない運用が推奨です。
- そのほか: `requirements.txt`（依存関係）, `.gitignore`, `directory_structure.txt` など

## キャリブレーションプロット（場所のメモ）

- スクリプト: `Analysis/src/evaluation/calibration_plots_from_odds.py`, `Analysis/src/evaluation/calibration_plots_sklearn.py`
- 出力先: `Analysis/results/calibration_plots/`, `Analysis/results/calibration_plots_sklearn/`

## セットアップ（簡易）

- 依存関係の導入: `pip install -r requirements.txt`
- Windows の場合は同梱の仮想環境 `boatrace/` を使わず、可能であれば新規の venv/conda 環境を作ることを推奨します。

## 変更・運用メモ

- 大きな生成物（例: 画像やDBファイル）は `Analysis/results/` や `Poseidon/` に出力されます。
- To-Do はこの README の先頭ブロックで管理できます。必要に応じて項目を追加・チェック・削除してください。
