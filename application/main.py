import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import Model
from AutoML import AutoMLExperiment

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TARGET_COLUMNS = {
     "bone tumor": "Treatment", 
     "Covid19" : "CLASIFFICATION_FINAL",
     #"Dengue": "Outcome", 
     #"harvests" : "GY",
     "Heart Attack" : "target",
     "heart disease" : "class",
     "heart failure" : "HeartDisease",
     #"nhanes" : "DIQ010",
     "parkinsons" : "status",
     "Thyroid" : "Recurred",
}

def preprocess_data(df, target_column):
    processed_df = df.copy()
    processed_df = processed_df.fillna(0)
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object' or processed_df[col].dtype == 'string':
            processed_df[col] = processed_df[col].fillna(0)
            
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            #print(f"Label Encoding applied to column: {col}")
    
    print(processed_df.head(3))
    return processed_df


def split_data(df, target_column, shot_size = 8, run_num = None):
    train_df, test_df = train_test_split(
        df,
        train_size=0.8,
        stratify=df[target_column],
        random_state=42
    )

    class_counts = train_df[target_column].value_counts()
    total_samples = sum(class_counts)
    
    # 각 클래스별 샘플 개수 계산 (비율 유지, 최소 1개 이상)
    samples_per_class = (class_counts / total_samples * shot_size).round().astype(int)
    samples_per_class = samples_per_class.clip(lower=1)  # 최소 1개 이상 유지

    # 실제 할당된 샘플 개수가 shot_size보다 클 경우, 비율을 다시 조정
    if samples_per_class.sum() > shot_size:
        factor = shot_size / samples_per_class.sum()
        samples_per_class = (samples_per_class * factor).round().astype(int)
        samples_per_class = samples_per_class.clip(lower=1)  # 최소 1개 유지

    # 각 클래스별 샘플링
    sampled_dfs = [train_df[train_df[target_column] == cls].sample(n=samples_per_class[cls], random_state=run_num) 
                   for cls in samples_per_class.index]
    train_shot_df = pd.concat(sampled_dfs)

    return train_shot_df, test_df


def main():
    pipeline = Model()
    
    n_features = 10           # Number of new features to generate
    n_top_columns = 5        # Number of top columns to consider
    shot_size = 8           # sample size for few-shot

    # Process each dataset
    for dataset_name, target_column in TARGET_COLUMNS.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Target column: {target_column}")
        
        try:
            # Load dataset
            dataset_path = os.path.join(DATA_DIR, dataset_name, f"{dataset_name}.csv")
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset file not found at {dataset_path}")
                continue
                
            df = pd.read_csv(dataset_path)
            
            # 5회 실험 반복
            for run_num in range(5, 6):  # 1부터 5까지
                print(f"\n실험 {run_num}/5 시작...")
                
                # 데이터 전처리 수행 (매 실험마다)
                processed_df = preprocess_data(df, target_column)
                
                # Train과 Test 데이터 분할 (매 실험마다)
                train_df, test_df = split_data(processed_df, target_column, shot_size, run_num)
                print(f"Train 데이터 크기: {len(train_df)} Test 데이터 크기: {len(test_df)}")
                
                # Run the feature engineering pipeline
                
                pipeline.run(
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    n_features=n_features,
                    n_top_columns=n_top_columns,
                    dataset_name=dataset_name
                )
                
                # AutoML 실험 실행
                print(f"AutoML 실험 {run_num} 시작...")
                automl_experiment = AutoMLExperiment(
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    dataset_name=dataset_name
                )
                automl_experiment.run_all(run_num)  # run_num 인자 전달
                
                print(f"실험 {run_num}/5 완료")
            
            print(f"{dataset_name} 처리 완료")
            print(f"결과 저장 위치: {os.path.join(OUTPUT_DIR, dataset_name)}")
            
        except Exception as e:
            print(f"{dataset_name} 처리 중 오류 발생: {str(e)}")
            continue