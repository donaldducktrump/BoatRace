import pandas as pd
import os
from get_data import load_fan_data


# ファンデータを保存する関数
def process_and_save_fan_data(fan_file):
    # Load fan data
    df_fan = load_fan_data(fan_file)

    # 指定された列を保存
    columns_to_add = ['選手登番','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    df_fan_filtered = df_fan[columns_to_add]

    # Define output directory for the fan dataframe
    output_dir = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe'
    os.makedirs(output_dir, exist_ok=True)

    # Save the filtered fan data
    output_file = os.path.join(output_dir, 'fan_data.csv')
    df_fan_filtered.to_csv(output_file, index=False)
    print(f"Fan data saved to: {output_file}")

def load_processed_fan_data():
    # Load the processed fan dataframe directly from saved CSV file
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe/fan_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

# ファンデータの保存
fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
process_and_save_fan_data(fan_file)
