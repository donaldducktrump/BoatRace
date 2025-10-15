import pandas as pd
import os
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, preprocess_data_old, load_before_data, load_before1min_data ,merge_before_data

# Define the months and corresponding dates for processing
month_folders = ['2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
date_files = {
    '2310': [f'2310{day:02d}' for day in range(1, 32)],
    '2311': [f'2311{day:02d}' for day in range(1, 31)],
    '2312': [f'2312{day:02d}' for day in range(1, 32)],
    '2401': [f'2401{day:02d}' for day in range(1, 32)],
    '2402': [f'2402{day:02d}' for day in range(1, 29)],
    '2403': [f'2403{day:02d}' for day in range(1, 32)],
    '2404': [f'2404{day:02d}' for day in range(1, 31)],
    '2405': [f'2405{day:02d}' for day in range(1, 32)],
    '2406': [f'2406{day:02d}' for day in range(1, 31)],
    '2407': [f'2407{day:02d}' for day in range(1, 32)],
    '2408': [f'2408{day:02d}' for day in range(1, 32)],
    '2409': [f'2409{day:02d}' for day in range(1, 31)],
    '2410': [f'2410{day:02d}' for day in range(1, 22)]
}
def remove_common_columns(df_left, df_right, on_columns):
    """
    df_leftとdf_rightのマージ時に、on_columnsを除く共通の列をdf_rightから削除する。
    """
    common_cols = set(df_left.columns).intersection(set(df_right.columns)) - set(on_columns)
    if common_cols:
        print(f"共通列: {common_cols}。df_rightからこれらの列を削除します。")
        df_right = df_right.drop(columns=common_cols)
    return df_right

def process_and_save_data():
    data_list_odds = []
    
    for month in month_folders:
        for date in date_files[month]:
            try:
                # Define file paths for each date
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                # Load the data files
                bdata_odds = load_b_file(b_file)
                before_data_odds = load_before_data(before_file)

                if not bdata_odds.empty and not before_data_odds.empty:
                    # Remove common columns and merge the datasets
                    before_data_odds = remove_common_columns(bdata_odds, before_data_odds, on_columns=['選手登番', 'レースID', '艇番'])
                    bdata_odds = merge_before_data(bdata_odds, before_data_odds)
                    data_list_odds.append(bdata_odds)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, または {before_file}")

    # If there is data to process
    if data_list_odds:
        data_odds = pd.concat(data_list_odds, ignore_index=True)
        data_odds = preprocess_data_old(data_odds)
        # Calculate 'win_odds_mean' based on '選手登番' and '艇番'
        data_odds['win_odds_mean'] = data_odds.groupby(['選手登番', '艇番'])['win_odds'].transform(lambda x: x.mean())
        selected_columns = ['選手登番', '艇番', 'win_odds', 'win_odds_mean']  # Add any additional columns you need
        data_odds = data_odds[selected_columns]

        # Define output directory for the processed dataframe
        output_dir = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/odds_dataframe'
        os.makedirs(output_dir, exist_ok=True)

        # Save the combined dataframe
        output_file = os.path.join(output_dir, 'odds_data.csv')
        data_odds.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
    else:
        print("No data to save.")

def load_processed_data():
    # Load the processed dataframe directly from saved CSV file
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/odds_dataframe/odds_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

# Run the function to process and save data
process_and_save_data()
