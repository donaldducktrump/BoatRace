import os
import pandas as pd
import wget
import urllib.request
import re
from lhafile import LhaFile
from tqdm import tqdm
from datetime import datetime, timedelta

# LZHファイルを解凍する関数
def unlha_file(lzh_path, extract_path=None):
    """
    LZHファイルを解凍する関数 
    ※日本語２バイト文字は解凍時にファイル名が文字化けします。また、ファイルが壊れる可能性があります。
    
    Args:
        lzh_path (str): LZHファイルのパス
        extract_path (str, optional): 解凍先のパス。未指定の場合はLZHファイルと同じ場所に解凍する。
        
    Returns:
        bool: 解凍が成功した場合はTrue、失敗した場合はFalseを返す。
    """
    
    try:
        # 解凍先パスが未指定の場合はLZHファイルと同じ場所に解凍する
        if extract_path is None:
            extract_path = os.path.dirname(lzh_path)
        
        # LZHファイルを開く
        lzh_ref = LhaFile(lzh_path)
        
        # 各ファイルを解凍
        for file_info in lzh_ref.infolist():
            file_path = os.path.join(extract_path, file_info.filename)
            
            if file_info.filename.endswith('/'):  # ディレクトリの場合
                os.makedirs(file_path, exist_ok=True) 
            else: 
                os.makedirs(os.path.dirname(file_path), exist_ok=True) 
                with open(file_path, 'wb') as output_file:
                    output_file.write(lzh_ref.read(file_info.filename))
        
        lzh_ref.fp.close()  # LZHファイルを閉じる

        return True  # 解凍成功時
    except Exception as e:
        print("Error during extraction:", str(e))
        return False  # 解凍失敗時

# 圧縮ファイルをウェブからダウンロードし解凍 >> テキストファイルを保存
def download_file(obj, date, base_dir, extract_dir):
    """
    obj (str): 'racelists' または 'results'
    date (str): 'YYYY-MM-DD' 形式の日付
    base_dir (str): ダウンロード先のベースディレクトリ
    extract_dir (str): 解凍先のベースディレクトリ
    """
    try:
        # 日付を 'YYYYMMDD' 形式に変換
        date_obj = pd.to_datetime(date).date()
        ymd = str(date_obj).replace('-', '')

        # 'K'/'k' または 'B'/'b' を設定
        S, s = ('K', 'k') if obj == 'results' else ('B', 'b')

        # 保存先ファイルパス
        txt_filepath = os.path.join(extract_dir, f'{ymd}.txt')

        # 既にファイルが存在する場合はスキップ
        if os.path.exists(txt_filepath):
            print(f"{obj} data for {ymd} already exists. Skipping download.")
            return
        else:
            # 保存先ディレクトリを作成（存在しない場合）
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(extract_dir, exist_ok=True)

            # ダウンロード先の.lzhファイルパス
            lzh_filepath = os.path.join(base_dir, f'{ymd}.lzh')

            # URLの構築
            url_t = f'http://www1.mbrace.or.jp/od2/{S}/'
            url_b = f'{ymd[:-2]}/{s}{ymd[2:]}.lzh'
            full_url = url_t + url_b

            try:
                # ファイルのダウンロード
                wget.download(full_url, lzh_filepath)
                print(f"\nDownloaded {full_url} to {lzh_filepath}")

                # LZHファイルの解凍
                extraction_success = unlha_file(lzh_filepath, extract_path=extract_dir)
                if extraction_success:
                    print(f"Extracted and saved to {txt_filepath}")
                    # 解凍後にファイルが存在するか確認
                    if not os.path.exists(txt_filepath):
                        print(f"Expected extracted file {txt_filepath} not found.")
                else:
                    print(f"Failed to extract {lzh_filepath}")

                # 圧縮ファイルの削除
                if os.path.exists(lzh_filepath):
                    os.remove(lzh_filepath)
                    print(f"Removed compressed file {lzh_filepath}")

            except urllib.error.HTTPError:
                print(f'There are no data for {date} ({obj}).')
            except Exception as e:
                print(f'An error occurred for {date} ({obj}): {e}')

    except Exception as e:
        print(f'Invalid date format or other error: {e}')

def generate_date_range(start_date, end_date):
    """
    指定された開始日から終了日までのリストを生成
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end - start
    return [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]

def main():
    # 開始合図
    print('作業を開始します')

    # ダウンロードする日付範囲
    start_date = pd.to_datetime('2024-11-09')
    end_date = pd.to_datetime('2024-11-09')
    date_range = pd.date_range(start=start_date, end=end_date)


    # 各日付に対して 'racelists' と 'results' をダウンロード
    for date in tqdm(date_range, desc="Downloading data"):
        # 年月情報を取得してディレクトリを作成
        year_month_str = date.strftime('%y%m')  # 'yyMM'形式にフォーマット
        racelists_dir = fr'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\b_data\{year_month_str}unlzh'
        results_dir = fr'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\k_data\{year_month_str}unlzh'
        racelists_extract_dir = fr'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\b_data\{year_month_str}'
        results_extract_dir = fr'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\k_data\{year_month_str}'

        # 必要なディレクトリが存在しない場合は作成
        os.makedirs(racelists_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(racelists_extract_dir, exist_ok=True)
        os.makedirs(results_extract_dir, exist_ok=True)

        date_str = date.strftime('%Y-%m-%d')

        # racelists のダウンロード
        download_file('racelists', date_str, racelists_dir, racelists_extract_dir)

        # results のダウンロード
        download_file('results', date_str, results_dir, results_extract_dir)

    # 終了合図
    print('作業を終了しました')

    # # 開始合図
    # print('作業を開始します')

    # # ダウンロードする日付範囲
    # start_date = '2024-10-20'
    # end_date = '2024-10-20'

    # # 日付リストの生成
    # date_list = generate_date_range(start_date, end_date)

    # # 保存先ディレクトリの定義
    # # ColabでGoogleドライブをマウントした状態を想定
    # # ローカル環境の場合は適宜パスを変更してください
    # racelists_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\b_data\2410unlzh'
    # results_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\k_data\2410unlzh'
    # racelists_extract_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\b_data\2410'
    # results_extract_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\k_data\2410'


    # # 各日付に対して 'racelists' と 'results' をダウンロード
    # for date in tqdm(date_list, desc="Downloading data"):
    #     # racelists のダウンロード
    #     download_file('racelists', date, racelists_dir, racelists_extract_dir)

    #     # results のダウンロード
    #     download_file('results', date, results_dir, results_extract_dir)

    # # 終了合図
    # print('作業を終了しました')

# スクリプトの実行
if __name__ == "__main__":
    main()
