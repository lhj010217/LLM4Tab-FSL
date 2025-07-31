import os
import pandas as pd
from sklearn.model_selection import train_test_split
from model import TabClassifier

# 실험 대상 데이터셋 및 타깃 컬럼 매핑
TARGET_COLUMNS = {
    #"bone tumor": "Treatment",
    "Covid19": "CLASIFFICATION_FINAL",
    #"Heart Attack": "target",
    #"heart disease": "class",
    #"heart failure": "HeartDisease",
    #"parkinsons": "status",
    #"Thyroid": "Recurred",
}

def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir     = os.path.join(base_dir, "datasets")
    results_dir = os.path.join(base_dir, "split_experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    records = []  # 개별 실행 결과 누적용 리스트

    for run in range(1, 6):
        print(f"Starting run {run}")
        for ds_name, tgt_col in TARGET_COLUMNS.items():
            raw_path = os.path.join(raw_dir, ds_name, f"ablation_{ds_name}.csv")
            if not os.path.exists(raw_path):
                print(f"  Skipped {ds_name}: raw file not found.")
                continue

            raw_df = pd.read_csv(raw_path)

            # 1) 20% train / 80% eval
            train20, eval80 = train_test_split(
                raw_df,
                train_size=0.8,
                stratify=raw_df[tgt_col],
                random_state=42 + run  # run마다 시드 변경
            )
            #model20 = TabClassifier(raw_df, train20, eval80, tgt_col)
            #model20.fit()
            #out20 = os.path.join(
            #    results_dir,
            #    f"{ds_name}_20pct_run{run}.csv"
            #)
            #model20.save_evaluation_result(out20)
            # 결과 로드 및 기록
            #df20 = pd.read_csv(out20)
            #df20["dataset"] = ds_name
            #df20["split"]   = "20pct"
            #df20["run"]     = run
            #records.append(df20)

            # 2) 16-shot few-shot 학습
            if len(eval80) >= 16:
                few_train, few_eval = train_test_split(
                    eval80,
                    train_size=16,
                    stratify=eval80[tgt_col],
                    random_state=42 + run
                )
            else:
                few_train = eval80.sample(n=16, replace=True, random_state=42 + run)
                few_eval = eval80.drop(few_train.index)

            model16 = TabClassifier(raw_df, few_train, few_eval, tgt_col)
            model16.fit()
            out16 = os.path.join(
                results_dir,
                f"{ds_name}_16shot_run{run}.csv"
            )
            model16.save_evaluation_result(out16)
            df16 = pd.read_csv(out16)
            df16["dataset"] = ds_name
            df16["split"]   = "16shot"
            df16["run"]     = run
            records.append(df16)

    # 누적된 결과를 하나의 DataFrame으로 통합
    all_df = pd.concat(records, ignore_index=True)

    # 데이터셋·분할 유형별 평균 계산
    avg_df = all_df.groupby(["dataset", "split"]).mean().reset_index()

    # 평균 결과 출력
    print("\n=== Average Metrics Across 5 Runs ===")
    print(avg_df.to_string(index=False))

if __name__ == "__main__":
    main()
