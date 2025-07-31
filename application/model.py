import pandas as pd
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import requests
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        
    def get_column_info(self) -> Dict[str, Any]:
        """Extract column names, data types, and example data"""
        column_info = {
            'columns': {},
            'target_column': self.target_column,
            'example': self.df.head(4).to_dict('records')[0]
        }
        
        for col in self.df.columns:
            column_info['columns'][col] = str(self.df[col].dtype)
            
        return column_info

class LLMInterface:
    def __init__(self):
        self.model_id = "llama3.1" # Back-bone model
        self.api_url = "http://localhost:11434/api/generate" # using Ollma
        
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the given prompt"""
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {str(e)}")
            
    def save_descriptions(self, task_desc: str, filepath: str) -> None:
        """Save task description to a text file, only including AI's responses"""
        # Convert list of dicts to string if necessary, only for AI responses
        if isinstance(task_desc, list):
            task_desc = '\n'.join(item['content'] for item in task_desc if isinstance(item, dict) and item.get('role') == 'assistant' and 'content' in item)
        
        with open(filepath, 'w') as f:
            f.write("[Task Description]\n")
            f.write(task_desc + "\n")
            
    def save_feature_code(self, feature_code: str, filepath: str) -> None:
        """Save generated feature code to a Python file"""
        with open(filepath, 'w') as f:
            f.write(feature_code)
            
    def get_task_description(self, column_info: Dict[str, Any]) -> str:
        prompt = self._create_prompt(column_info)
        return self._call_ollama(prompt)
    
    def _create_prompt(self, column_info: Dict[str, Any]) -> str:
        columns_info = "\n".join([
            f"- {col}: {dtype}" 
            for col, dtype in column_info['columns'].items()
        ])
        example_data = "\n".join([
            f"- {col}: {val}" 
            for col, val in column_info['example'].items()
        ])
        
        prompt = f"""<s>[INST] You are a highly experienced data scientist with expertise in feature engineering and predictive modeling.
        Analyze this dataset and provide the following information:

        1. Task Analysis:
        - Identify if this is a classification or regression problem
        - Describe the specific prediction objective
        

        2. Comprehensive Feature Analysis:
        For each feature in the dataset, please provide:
        - Domain knowledge and significance of the feature
        - Expected relationship with the target variable ({column_info['target_column']})
        - Potential correlation direction (positive/negative) and strength

        3. Feature Interactions:
        - Identify important feature pairs that might have synergistic effects
        - Suggest potential interaction terms that could be valuable

        Dataset Information:
        Target Variable: {column_info['target_column']}

        Available Features:
        {columns_info}

        Sample Data Point:
        {example_data}

        Provide a comprehensive analysis focusing on feature importance and potential feature engineering opportunities. [/INST]"""
        return prompt

class AttentionCalculator:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def calculate_attention_scores(self, 
                                 column_names: List[str], 
                                 description_file: str) -> Dict[str, float]:
        """Calculate attention scores for each column using BERT attention"""
        # Load description as text
        with open(description_file, 'r') as f:
            descriptions = f.read()
        
        # Process descriptions text to extract task description
        task_desc = descriptions.split('[Task Description]\n')[1].split('[Feature Description]\n')[0].strip()
        
        # Ensure task_desc is a string
        if not isinstance(task_desc, str):
            task_desc = str(task_desc)
        
        if not task_desc:
            raise ValueError("Task description is empty or invalid")
            
        # Calculate scores for each column using Query-Key-Value structure
        scores = {}
        # Get embeddings for task description
        task_desc_inputs = self.tokenizer(
            task_desc,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        task_desc_inputs = {k: v.to(self.device) for k, v in task_desc_inputs.items()}

        with torch.no_grad():
            task_desc_outputs = self.model(**task_desc_inputs)
            # Remove padding tokens and calculate mean
            task_desc_mask = task_desc_inputs['attention_mask'].bool()
            Q = task_desc_outputs.last_hidden_state[task_desc_mask].mean(dim=0)  # Query
            # Normalize the query vector
            Q = Q / Q.norm(dim=0, keepdim=True)

        for column in column_names:
            # Get embeddings for column name
            column_inputs = self.tokenizer(
                column,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            column_inputs = {k: v.to(self.device) for k, v in column_inputs.items()}

            with torch.no_grad():
                column_outputs = self.model(**column_inputs)
                # Remove padding tokens and calculate mean
                column_mask = column_inputs['attention_mask'].bool()
                K = column_outputs.last_hidden_state[column_mask].mean(dim=0)  # Key
                # Normalize the key vector
                K = K / K.norm(dim=0, keepdim=True)

            # Calculate attention score using dot product
            score = torch.matmul(Q, K.unsqueeze(1)).item()  # Ensure K is 2D
            scores[column] = score

        # Apply softmax to convert scores to probabilities
        total_score = sum(scores.values())
        scores = {k: v / total_score for k, v in scores.items()}

        return scores
    
    def get_top_columns(self, 
                       attention_scores: Dict[str, float], 
                       n_columns: int) -> List[str]:
        """Get top n columns based on attention scores"""
        sorted_columns = sorted(attention_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        return [col for col, _ in sorted_columns[:n_columns]]

class FeatureGenerator:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        
    def generate_feature_code(self, 
                            selected_columns: List[str], 
                            example_data: Dict[str, Any],
                            n_features: int,
                            dataset_name: str) -> str:
        """Generate Python code for new features using LLM"""
        # Load descriptions from file
        description_filepath = os.path.join(OUTPUT_DIR, dataset_name, 'task_description.txt')
        with open(description_filepath, 'r') as f:
            descriptions = f.read()
        
        # Create a prompt for the LLM
        prompt = f"""[INST]You are an experienced Python developer. Generate a Python function that follows the code template below. 
        Indicate the reason for generating features in the code as comments.
        
        Task and Feature description:
        {descriptions}
        
        [IMPORTANT RULES]
        The function should take a DataFrame 'df' as input and create exactly "{n_features} new features".
        new features focus on the following columns: {', '.join(selected_columns)}.
        Example data: {example_data}
        
        function name should be 'add_new_features', return the modified DataFrame.
        never drop the columns, only add new features

        you can only use libraries from pandas, numpy

        code template:
        ```python
        import pandas as pd
        import numpy as np

        def add_new_features(df):
            # 1. feature 1
            # reason: 
            # code

            ...and so on...

            # 10. feature 10
            # reason: 
            # code

            return df
        ```
        [/IMPORTANT RULES]
        [/INST]
        """
        
        python_code = self.llm_interface._call_ollama(prompt)
        
        # 코드 파싱
        if "```python" in python_code:
            # 코드 블록이 있는 경우
            code_start = python_code.find("```python") + len("```python")
            code_end = python_code.find("```", code_start)
            python_code = python_code[code_start:code_end].strip()
        
        # import 문과 함수 부분 추출
        if "def add_new_features(df):" in python_code:
            # import 문 추출
            import_lines = []
            for line in python_code.split('\n'):
                if line.strip().startswith('import '):
                    import_lines.append(line.strip())
            
            # 함수 부분 추출
            func_start = python_code.find("def add_new_features(df):")
            return_start = python_code.find("return", func_start)
            if return_start != -1:
                return_end = python_code.find("\n", return_start)
                if return_end == -1:
                    return_end = len(python_code)
                func_code = python_code[func_start:return_end + 1].strip()
                
                # import 문과 함수 코드 결합
                if import_lines:
                    python_code = '\n'.join(import_lines) + '\n\n' + func_code
                else:
                    python_code = func_code
            else:
                # return 문이 없는 경우 기본 템플릿 제공
                python_code = "import pandas as pd\nimport numpy as np\n\ndef add_new_features(df):\n    return df"
        else:
            # 함수 정의가 없는 경우 기본 템플릿 제공
            python_code = "import pandas as pd\nimport numpy as np\n\ndef add_new_features(df):\n    return df"
            
        return python_code
    
    def apply_feature_code(self, 
                          df: pd.DataFrame, 
                          feature_code_filepath: str) -> pd.DataFrame:
        """Apply generated feature code from a file to DataFrame"""
        # Read the feature code from the file
        with open(feature_code_filepath, 'r') as f:
            feature_code = f.read()
        
        # Create a new namespace for the function
        namespace = {}
        
        # Execute the feature code in the namespace
        exec(feature_code, namespace)
        
        # Get the function from the namespace
        add_new_features = namespace.get('add_new_features')
        if not add_new_features:
            raise ValueError("Function 'add_new_features' not found in the generated code")
        
        # Execute the function with the DataFrame
        return add_new_features(df)

class Model:
    def __init__(self):
        self.data_analyzer = None
        self.llm_interface = LLMInterface()
        self.attention_calculator = AttentionCalculator()
        self.feature_generator = FeatureGenerator(self.llm_interface)
        
    def run(self, 
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            target_column: str, 
            n_features: int, 
            n_top_columns: int,
            dataset_name: str) -> None:
        """Run the complete feature engineering pipeline"""
        # Initialize data analyzer with train data only
        self.data_analyzer = DataAnalyzer(train_df, target_column)
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Step 1: Get column information from train data
        column_info = self.data_analyzer.get_column_info()
        
        # Step 2: Get and save descriptions
        task_desc = self.llm_interface.get_task_description(column_info)
        description_filepath = os.path.join(dataset_output_dir, 'task_description.txt')
        self.llm_interface.save_descriptions(task_desc, description_filepath)
        
        # Step 3: Calculate attention scores using train data only
        # target column을 제외한 컬럼 리스트 생성
        feature_columns = [col for col in column_info['columns'].keys() if col != target_column]
        
        attention_scores = self.attention_calculator.calculate_attention_scores(
            feature_columns,
            description_filepath
        )
        print(f"Attention scores: {attention_scores}")

        # Step 4: Select top columns and generate feature code
        top_columns = self.attention_calculator.get_top_columns(
            attention_scores, 
            n_top_columns
        )
        print(f"Selected columns by attention scores: {top_columns}")

        feature_code = self.feature_generator.generate_feature_code(
            top_columns,
            column_info['example'],
            n_features,
            dataset_name
        )
        
        # Save generated feature code to a file
        feature_code_filepath = os.path.join(dataset_output_dir, 'feature_code.py')
        self.llm_interface.save_feature_code(feature_code, feature_code_filepath)
        
        # Step 5: Apply feature generation to both train and test data
        train_result_df = self.feature_generator.apply_feature_code(train_df, feature_code_filepath)
        test_result_df = self.feature_generator.apply_feature_code(test_df, feature_code_filepath)
        
        # Save enhanced datasets separately
        train_output_path = os.path.join(dataset_output_dir, 'Feature_Engineered_train.csv')
        test_output_path = os.path.join(dataset_output_dir, 'Feature_Engineered_test.csv')
        
        train_result_df.to_csv(train_output_path, index=False)
        test_result_df.to_csv(test_output_path, index=False)
        
        print(f"Enhanced train dataset saved to: {train_output_path}")
        print(f"Enhanced test dataset saved to: {test_output_path}")
        
        return
    

from tensorflow.keras.regularizers import l2
class TabClassifier():
    def __init__(self, raw_df, train_df, test_df, target_column):
        self.raw_df = self.preprocess_data(raw_df, target_column)
        self.train_df = self.preprocess_data(train_df, target_column)
        self.test_df = self.preprocess_data(test_df, target_column)
        self.target_column = target_column
        self.results = {}
        self.metric = None  # 평가 결과 저장

        # 결과 저장을 위한 디렉토리 생성
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "mymodel_output")
        self.output_dir = os.path.join(self.base_dir, "mymodel_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, y_true, y_pred):
        unique_classes = len(np.unique(y_true))
        f1_average = 'weighted' if unique_classes > 2 else 'binary'
            
        metrics = {
            'f1': f1_score(y_true, y_pred, average=f1_average),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        self.metric = metrics  # self.metric에 저장
        return metrics

    def preprocess_data(self, df, target_column):
        processed_df = df.copy()
        processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in processed_df.columns:
            if processed_df[col].dtype in ['object', 'string']:
                processed_df[col] = processed_df[col].fillna(0)
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col])
        return processed_df    

    def build_neural_network(self, input_dim, output_dim):
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization()
        ])
        
        if output_dim > 2:
            model.add(Dense(output_dim, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        else:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=0.007),
            loss=loss,
            metrics=['accuracy']
        )
        return model, loss
    

    def fit(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        raw_data = self.raw_df.drop(columns=[self.target_column])
        train_features = self.train_df.drop(columns=[self.target_column])
        extra_columns = [col for col in train_features.columns if col not in raw_data.columns]

        predictions = []
        for i in range(20):
            print(f"모델 {i+1}/20 학습 중...")
            print(f"Extra columns: {extra_columns}")
            selected_columns = np.random.choice(extra_columns, size=int(len(extra_columns) * 0.30), replace=False)
            raw_columns = [col for col in self.raw_df.columns if col != self.target_column]
            columns_to_use = list(selected_columns) + raw_columns
            
            X_train = self.train_df[columns_to_use]
            y_train = self.train_df[self.target_column]
            X_test = self.test_df[columns_to_use]
            print("train size : ", X_train.shape)
            print("test size : ", X_test.shape)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            num_classes = len(np.unique(y_train))
            model, loss_type = self.build_neural_network(X_train_scaled.shape[1], num_classes)
            
            model.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=16,
                verbose=1
            )
            
            if num_classes > 2:
                pred_proba = model.predict(X_test_scaled)
                predictions.append(pred_proba)
            else:
                pred_single = model.predict(X_test_scaled).flatten()
                pred_proba = np.column_stack((1 - pred_single, pred_single))
                predictions.append(pred_proba)
        
        print("모든 모델 학습 완료, 앙상블 결과 계산 중...")
        avg_predictions = np.mean(predictions, axis=0)
        final_predictions = np.argmax(avg_predictions, axis=1)

        metrics = self.evaluate(self.test_df[self.target_column], final_predictions)
        print("평가 결과:", metrics)
        
        return self

    def predict(self, test_df):
        pass

    def save_evaluation_result(self, path):
        if self.metric is None:
            print("평가 결과가 없습니다.")
            return
        
        metrics_df = pd.DataFrame([self.metric])
        metrics_df.to_csv(path, index=False)
        print(f"평가 결과가 {path}에 저장되었습니다.")
