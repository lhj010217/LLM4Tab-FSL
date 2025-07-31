import os
import pandas as pd
from model import TabClassifier
from sklearn.model_selection import train_test_split
TARGET_COLUMNS = {
    "bone tumor": "Treatment", 
    "Covid19": "CLASIFFICATION_FINAL",
    #"Dengue": "Outcome", 
    #"harvests": "GY",
    "Heart Attack": "target",
    "heart disease": "class",
    #"heart failure": "HeartDisease",
    #"nhanes": "DIQ010",
    "parkinsons": "status",
    "Thyroid": "Recurred",
}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "mymodel_output")
    results_dir = os.path.join(base_dir, "mymodel_results")
    os.makedirs(results_dir, exist_ok=True) 

    for run in range(1, 6):  
        print(f"Starting run {run}")
        
        for dataset_name, target_column in TARGET_COLUMNS.items():
            print(f"Processing dataset: {dataset_name} (Run {run})")
            
            dataset_path = os.path.join(base_dir, "datasets", dataset_name, f"{dataset_name}.csv")
            train_path = os.path.join(data_dir, dataset_name, "Feature_Engineered_train.csv")
            test_path = os.path.join(data_dir, dataset_name, "Feature_Engineered_test.csv")
            
            if not (os.path.exists(dataset_path) and os.path.exists(train_path) and os.path.exists(test_path)):
                print(f"Skipping {dataset_name}: Missing required files.")
                continue

            raw_df = pd.read_csv(dataset_path)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df, _ = train_test_split(
                    train_df,
                    train_size=16,
                    stratify=train_df[target_column],
                    random_state=42
                )
            model = TabClassifier(raw_df, train_df, test_df, target_column)
            model.fit()

            output_file = os.path.join(results_dir, f"result_{dataset_name}_{run}_shot8.csv")
            model.save_evaluation_result(output_file)
            print(f"Finished processing {dataset_name}, results saved to {output_file}")

if __name__ == "__main__":
    main()