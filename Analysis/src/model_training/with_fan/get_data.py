import pandas as pd

def parse_racer_data(line_bytes):
    # フィールドのバイト位置を定義（開始位置、終了位置）
    positions = {}
    pos = 0

    # 基本情報のフィールド
    basic_fields = [
        ('登番', 4),
        ('名前漢字', 16),
        ('名前カナ', 15),
        ('支部', 4),
        ('級', 2),
        ('年号', 1),
        ('生年月日', 6),
        ('性別', 1),
        ('年齢', 2),
        ('身長', 3),
        ('体重', 2),
        ('血液型', 2),
        ('勝率', 4),
        ('複勝率', 4),
        ('1着回数', 3),
        ('2着回数', 3),
        ('出走回数', 3),
        ('優出回数', 2),
        ('優勝回数', 2),
        ('平均スタートタイミング', 3),
    ]

    for field_name, field_size in basic_fields:
        positions[field_name] = (pos, pos + field_size)
        pos += field_size

    # コース別のフィールド（進入回数、複勝率、平均STタイミング、平均ST順位）
    course_fields = ['進入回数', '複勝率', '平均スタートタイミング', '平均スタート順位']
    for i in range(1, 7):
        for field_name in course_fields:
            full_field_name = f'{i}コース{field_name}'
            field_size = 3 if '進入回数' in field_name or '平均スタートタイミング' in field_name or '平均スタート順位' in field_name else 4
            positions[full_field_name] = (pos, pos + field_size)
            pos += field_size

    # ランク・能力指数などのフィールド
    additional_fields = [
        ('前期級', 2),
        ('前々期級', 2),
        ('前々々期級', 2),
        ('前期能力指数', 4),
        ('今期能力指数', 4),
        ('年', 4),
        ('期', 1),
        ('算出期間（自）', 8),
        ('算出期間（至）', 8),
        ('養成期', 3),
    ]
    for field_name, field_size in additional_fields:
        positions[field_name] = (pos, pos + field_size)
        pos += field_size

    # 各コースの詳細フィールド
    detailed_course_fields = [
        ('1着回数', 3),
        ('2着回数', 3),
        ('3着回数', 3),
        ('4着回数', 3),
        ('5着回数', 3),
        ('6着回数', 3),
        ('F回数', 2),
        ('L0回数', 2),
        ('L1回数', 2),
        ('K0回数', 2),
        ('K1回数', 2),
        ('S0回数', 2),
        ('S1回数', 2),
        ('S2回数', 2),
    ]

    for i in range(1, 7):
        for field_name, field_size in detailed_course_fields:
            full_field_name = f'{i}コース{field_name}'
            positions[full_field_name] = (pos, pos + field_size)
            pos += field_size

    # コースなしのフィールド
    no_course_fields = [
        ('コースなしL0回数', 2),
        ('コースなしL1回数', 2),
        ('コースなしK0回数', 2),
        ('コースなしK1回数', 2),
    ]
    for field_name, field_size in no_course_fields:
        positions[field_name] = (pos, pos + field_size)
        pos += field_size

    # 出身地（6バイト）
    positions['出身地'] = (pos, pos + 6)
    pos += 6

    data = {}
    for field, (start, end) in positions.items():
        field_bytes = line_bytes[start:end]
        # デコードして文字列に変換（エラーを無視）
        value = field_bytes.decode('cp932', errors='ignore').strip()
        data[field] = value

    return data

def load_fan_data(file_path):
    # ファンデータを読み込んで解析し、データフレームを返す関数
    def parse_racer_data(line_bytes):
        positions = {}
        pos = 0

        # 基本情報のフィールド
        basic_fields = [
            ('選手登番', 4),
            ('名前漢字', 16),
            ('名前カナ', 15),
            ('支部', 4),
            ('級', 2),
            ('年号', 1),
            ('生年月日', 6),
            ('性別', 1),
            ('年齢', 2),
            ('身長', 3),
            ('体重', 2),
            ('血液型', 2),
            ('勝率', 4),
            ('複勝率', 4),
            ('1着回数', 3),
            ('2着回数', 3),
            ('出走回数', 3),
            ('優出回数', 2),
            ('優勝回数', 2),
            ('平均スタートタイミング', 3),
        ]

        for field_name, field_size in basic_fields:
            positions[field_name] = (pos, pos + field_size)
            pos += field_size

        # コース別のフィールド（進入回数、複勝率、平均STタイミング、平均ST順位）
        course_fields = ['進入回数', '複勝率', '平均スタートタイミング', '平均スタート順位']
        for i in range(1, 7):
            for field_name in course_fields:
                full_field_name = f'{i}コース{field_name}'
                field_size = 3 if '進入回数' in field_name or '平均スタートタイミング' in field_name or '平均スタート順位' in field_name else 4
                positions[full_field_name] = (pos, pos + field_size)
                pos += field_size

        # ランク・能力指数などのフィールド
        additional_fields = [
            ('前期級', 2),
            ('前々期級', 2),
            ('前々々期級', 2),
            ('前期能力指数', 4),
            ('今期能力指数', 4),
            ('年', 4),
            ('期', 1),
            ('算出期間（自）', 8),
            ('算出期間（至）', 8),
            ('養成期', 3),
        ]
        for field_name, field_size in additional_fields:
            positions[field_name] = (pos, pos + field_size)
            pos += field_size

        # 各コースの詳細フィールド
        detailed_course_fields = [
            ('1着回数', 3),
            ('2着回数', 3),
            ('3着回数', 3),
            ('4着回数', 3),
            ('5着回数', 3),
            ('6着回数', 3),
            ('F回数', 2),
            ('L0回数', 2),
            ('L1回数', 2),
            ('K0回数', 2),
            ('K1回数', 2),
            ('S0回数', 2),
            ('S1回数', 2),
            ('S2回数', 2),
        ]

        for i in range(1, 7):
            for field_name, field_size in detailed_course_fields:
                full_field_name = f'{i}コース{field_name}'
                positions[full_field_name] = (pos, pos + field_size)
                pos += field_size

        # コースなしのフィールド
        no_course_fields = [
            ('コースなしL0回数', 2),
            ('コースなしL1回数', 2),
            ('コースなしK0回数', 2),
            ('コースなしK1回数', 2),
        ]
        for field_name, field_size in no_course_fields:
            positions[field_name] = (pos, pos + field_size)
            pos += field_size

        # 出身地（6バイト）
        positions['出身地'] = (pos, pos + 6)
        pos += 6

        data = {}
        for field, (start, end) in positions.items():
            field_bytes = line_bytes[start:end]
            # デコードして文字列に変換（エラーを無視）
            value = field_bytes.decode('cp932', errors='ignore').strip()
            data[field] = value

        return data

    data_list = []

    try:
        with open(file_path, 'rb') as file:
            lines = [line.rstrip(b'\n') for line in file if line.strip()]
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame()

    # 各行を処理
    for line_bytes in lines:
        if len(line_bytes) < 416:
            print(f"行の長さが不足しています（{len(line_bytes)}バイト）: {line_bytes}")
            continue
        racer_dict = parse_racer_data(line_bytes)
        if racer_dict:
            data_list.append(racer_dict)
        else:
            print(f"行のパースに失敗しました: {line_bytes}")

    # DataFrameに変換
    df_fan = pd.DataFrame(data_list)

    # 非数値データの列を指定
    non_numeric_columns = [
        '選手登番', '名前漢字', '名前カナ', '支部', '級', '血液型', '年号',
        '前期級', '前々期級', '前々々期級', '出身地'
    ]

    # 数値データの列を特定
    numeric_columns = [col for col in df_fan.columns if col not in non_numeric_columns]

    # 数値データの列を数値型に変換
    for col in numeric_columns:
        df_fan[col] = pd.to_numeric(df_fan[col], errors='coerce')

    return df_fan

def add_course_data(row, df_fan):
    # 各行に対して、コース別データを追加する関数
    racer_id = row['選手登番']
    boat_number = row['艇番']  # '艇番'は1～6の整数と仮定

    # 該当する選手のファンデータを取得
    racer_fan_data = df_fan[df_fan['選手登番'] == racer_id]
    if racer_fan_data.empty:
        return row  # データがない場合はそのまま返す

    # '艇番'に対応するコース番号を取得
    course_num = str(int(boat_number))
    course_columns = [
        '進入回数', '複勝率', '平均スタートタイミング', '平均スタート順位',
        '1着回数', '2着回数', '3着回数', '4着回数', '5着回数', '6着回数',
        'F回数', 'L0回数', 'L1回数', 'K0回数', 'K1回数', 'S0回数', 'S1回数', 'S2回数'
    ]

    for col in course_columns:
        fan_col_name = f'{course_num}コース{col}'
        if fan_col_name in df_fan.columns:
            row[fan_col_name] = racer_fan_data.iloc[0][fan_col_name]
        else:
            row[fan_col_name] = None  # データがない場合はNoneを設定

    return row

def main():
    # ファイルパスを指定
    file_path = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'

    data_list = []

    try:
        with open(file_path, 'rb') as file:
            lines = [line.rstrip(b'\n') for line in file if line.strip()]
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 各行を処理
    for line_bytes in lines:
        if len(line_bytes) < 416:
            print(f"行の長さが不足しています（{len(line_bytes)}バイト）: {line_bytes}")
            continue
        racer_dict = parse_racer_data(line_bytes)
        if racer_dict:
            data_list.append(racer_dict)
        else:
            print(f"行のパースに失敗しました: {line_bytes}")

    # DataFrameに変換
    df = pd.DataFrame(data_list)

    # 非数値データの列を指定
    non_numeric_columns = [
        '名前漢字', '名前カナ', '支部', '級', '血液型', '年号',
        '前期級', '前々期級', '前々々期級', '出身地'
    ]

    # 数値データの列を特定
    numeric_columns = [col for col in df.columns if col not in non_numeric_columns]

    # 数値データの列を数値型に変換
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    pd.set_option('display.max_columns', 100)
    print(df.head())
    # 結果を表示
    print(df[['登番', '名前漢字', '前期級', '前々期級', '前々々期級','算出期間（自）','算出期間（至）']].head())

    # 必要に応じて、データをCSVファイルに保存
    # df.to_csv('fan_data.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
