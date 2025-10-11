import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.features import preprocess_for_training

def prepare_for_lstm(df):
    '''
    Hàm này tách X, y, cân bằng SMOTE, chia train/test và rehape.
    '''
    print("Preparing data for LSTM...")

    # 1. Tách đặc trưng X và nhãn y.
    X = df.drop('Label', axis = 1).values
    y = df['Label'].values

    # 2. Cân bằng dữ liệu bằng SMOTE.
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 3. Chuyển đổi nhãn sang dạng categorical (one-hot encoding).
    # để sử dụng cho hàm loss "categorical_crossentropy".
    num_classes = len(np.unique(y_resampled))
    y_resampled_cat = to_categorical(y_resampled)

    # 4. Chia dữ liệu thành tập train và tập test.
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled_cat, 
        test_size = 0.2, 
        random_state=42, 
        stratify=y_resampled # Giữ tỷ lệ phân phối nhãn trong cả hai tập.
    )

    # 5. Reshape dữ liệu cho đầu vào của lstm (samples, timestep, features).
    # mỗi mẫu là một chuỗi có 1 bước thời gian timestep
    timestep = 1
    features = X_train.shape[1]
    X_train = X_train.reshape(X_train.shape[0], timestep, features)
    X_test = X_test.reshape(X_test.shape[0], timestep, features)

    print("Data preparation complete.")
    return X_train, X_test, y_train, y_test

def create_lstm_model(input_shape, num_classes):
    '''
    Hàm này định nghĩa kiến trúc của model LSTM.
    '''
    print("Creating LSTM model...")
    model = Sequential([
        Input(shape = input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax') # Lớp output với softmax cho phân loại nhiều lớp.
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy', 
                metrics='accuracy'
    )
    print("Model created and compiled.")
    model.summary() # In ra cấu trúc của model.
    return model

# PHẦN THỰC THI CHÍNH CHO MODEL
if __name__ == '__main__':
    # 1. Cấu hình.
    RAW_DATA_PATH = "../data/raw/CSE-CIC-IDS2018"
    MODEL_SAVA_PATH = "../models/model.h5"

    # 2. Tải và tiền xử lý dữ liệu.
    all_files = [os.path.join(RAW_DATA_PATH, f) for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]

    processed_frames = []
    SAMPLE_SIZE = 50000

    print(f"Start processing {len(all_files)} file...")

    # Xử lý từng file
    for file in all_files:
        try:
            df = pd.read_csv(file)

            if len(file) > SAMPLE_SIZE:
                processed_frames.append(df.sample(n=SAMPLE_SIZE, random_state=42))
            else:
                processed_frames.append(df)

            print(f"Completed processing: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error read file {file}: {e}")
    
    # Ghép nối tất cả các dataframe trong danh sách lại.
    print("Concatenating dataframes...")
    if processed_frames:
        combined_df = pd.concat(processed_frames, ignore_index=True)
        print("Completed!")

        combined_df.info()
        print(combined_df.head())
    else:
        print("No files are processed!")
    
    print("Processing data...")
    processed_df = preprocess_for_training(combined_df)

    # 3. Chuẩn bị dữ liệu cho model
    X_train, X_test, y_train, y_test = prepare_for_lstm(processed_df)

    # 4. Xây dựng và huấn luyện model.
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    model = create_lstm_model(input_shape=input_shape, num_classes=num_classes)

    # Thêm callbacks để chống overfitting và lưu model tốt nhất.
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        restore_best_weights = True
    )
    model_checkpoint = ModelCheckpoint(
        MODEL_SAVA_PATH,
        monitor='val_accuracy',
        save_best_only=True
    )

    print("Training the model...")
    model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs = 25,
        batch_size = 64,
        callbacks = [early_stopping, model_checkpoint]
    )

    print("Training process finished!")
