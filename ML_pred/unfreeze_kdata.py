import os
import shutil

# 移動元ディレクトリのベース
input_dir_base = r"C:\\Users\\kazut\\OneDrive\\デスクトップ\\"

# 移動先ディレクトリ
output_dir = r"C:\\Users\\kazut\\OneDrive\\Documents\\BoatRace\\ML_pred\\k_data\\2407"

# 移動元ディレクトリに保存されているb_dataの範囲
start_date = 240701
end_date = 240731

# 移動処理
def move_b_files(input_base, output_directory, start, end):
    for date in range(start, end + 1):
        # フォルダ名とファイル名を作成
        folder_name = f"k{date}"
        file_name = f"K{date}.TXT"
        
        # 移動元のパス
        input_folder_path = os.path.join(input_base, folder_name)
        input_file_path = os.path.join(input_folder_path, file_name)
        print(f"ファイルの存在確認: {input_file_path}")
        
        # 移動先のディレクトリを作成（存在しない場合のみ）
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # ファイルの存在確認
        if os.path.exists(input_file_path):
            # ファイルを移動
            shutil.move(input_file_path, os.path.join(output_directory, file_name))
            print(f"{file_name} を {output_directory} に移動しました。")
        else:
            print(f"{input_file_path} が見つかりません。")
# メイン処理
if __name__ == "__main__":
    move_b_files(input_dir_base, output_dir, start_date, end_date)
