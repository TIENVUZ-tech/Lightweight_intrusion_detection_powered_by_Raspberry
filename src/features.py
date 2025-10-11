import pandas as pd
import numpy as np
import sklearn.preprocessing as MinMaxScaler, LabelEncoder
import joblib # dùng để lưu các đối tượng scaler và encoder

# Đường dẫn để lưu các object tiền xử lý.
SCALER_PATH = '../models/scaler.pkl'
ENCODER_PATH = '../models/label_encoder.pkl'

def preprocess_for_training(combined_df):
    '''
    Thực hiện quá trình tiền xử lý dữ liệu cho training.
    '''
    # 1. Xóa các cột không cần thiết.
    columns_to_drop = ['Dst Port', 'Pkt Len Var', 'Pkt Size Avg', 'Fwd Seg Size Avg', 
                       'Bwd Seg Size Avg', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd PSH Flags', 
                       'CWE Flag Count', 'ECE Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 
                       'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 
                       'Fwd Seg Size Min']
    combined_df.drop(columns = columns_to_drop, axis = 1, inplace = True, error = 'ignore')

    # 2. Xử lý giá trị thiếu cho các cột số.
    numeric_columns = combined_df.select_dtypes(include = np.number).columns
    combined_df[numeric_columns] = combined_df[numeric_columns].fillna(combined_df[numeric_columns].median())

    # 3. Mã hóa nhãn.
    label_encoder = LabelEncoder()
    combined_df['Label'] = label_encoder.fit_transform(combined_df['Label'])
    # Lưu lại encoder để dùng sau này.
    joblib.dump(label_encoder, ENCODER_PATH)

    # 4. Chuẩn hóa dữ liệu.
    features = combined_df.drop('Label', axis = 1)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    # Lưu lại scaler để dùng cho dữ liệu mới.
    joblib.dump(scaler, SCALER_PATH)

    # Gắn lại vào datafram
    combined_df[features.columns] = scaled_features

    print("Processing for training completed!")
    return combined_df

def preprocessing_for_inference(data_input):
    '''
    Thực hiện quá trình xử lý dữ liệu mới (live traffic) trước khi dự đoán.
    '''
    # Nạp sacler từ quá trình traning.
    scaler = joblib.load(SCALER_PATH)

    # Áp dụng chuẩn hóa.
    scaled_data = scaler.transform(data_input)

    return scaled_data