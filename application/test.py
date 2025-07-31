import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
import autogluon.tabular as ag
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from supervised import AutoML
import warnings
import time
from copy import deepcopy
warnings.filterwarnings('ignore')

class AutoMLExperiment:
    def __init__(self, train_df, test_df, target_column, dataset_name, n_repetitions=3):
        self.train_df = train_df
        self.test_df = test_df
        self.target_column = target_column
        self.dataset_name = dataset_name
        self.n_repetitions = n_repetitions
        self.results = {}
        
        # 결과 저장을 위한 디렉토리 생성
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "automl_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, y_true, y_pred):
        """평가 지표 계산"""
        # 고유한 클래스 수 확인
        unique_classes = len(np.unique(y_true))
        
        # 다중 분류인 경우 average 파라미터 설정
        if unique_classes > 2:
            f1_average = 'weighted'
        else:
            f1_average = 'binary'
            
        metrics = {
            'f1': f1_score(y_true, y_pred, average=f1_average),
            'accuracy': accuracy_score(y_true, y_pred)
        }
            
        return metrics
    
    def run_h2o(self, repetition_idx):
        """H2O AutoML 실행"""
        print(f"\nRunning H2O AutoML (Repetition {repetition_idx+1}/{self.n_repetitions})...")
        h2o.init()
        
        # 데이터 변환
        train_h2o = h2o.H2OFrame(self.train_df)
        test_h2o = h2o.H2OFrame(self.test_df)
        
        # 실행마다 다른 seed 사용
        seed = 42 + repetition_idx
        
        # AutoML 실행
        aml = H2OAutoML(seed=seed, max_runtime_secs=300)
        aml.train(x=list(set(self.train_df.columns) - {self.target_column}),
                 y=self.target_column,
                 training_frame=train_h2o)
        
        # 예측
        pred = aml.predict(test_h2o)
        y_pred_proba = pred['predict'].as_data_frame().values
        y_pred = np.argmax(y_pred_proba, axis=1)
        # 평가
        metrics = self.evaluate(
            self.test_df[self.target_column],
            y_pred
        )
        
        h2o.cluster().shutdown()
        return metrics
    
    def run_tpot(self, repetition_idx):
        """TPOT 실행"""
        print(f"\nRunning TPOT (Repetition {repetition_idx+1}/{self.n_repetitions})...")
        
        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        # 실행마다 다른 seed 사용
        seed = 42 + repetition_idx
        
        # TPOT 실행
        tpot = TPOTClassifier(random_state=seed, 
                              scoring='f1_weighted',
                              max_time_mins=5, 
                              n_jobs=-1)
        tpot.fit(X_train, y_train)
        
        # 예측
        y_pred = tpot.predict(X_test)
        
        # 평가
        return self.evaluate(y_test, y_pred)
    
    def run_autogluon(self, repetition_idx):
        """AutoGluon 실행"""
        print(f"\nRunning AutoGluon (Repetition {repetition_idx+1}/{self.n_repetitions})...")
        
        # 데이터 준비
        train_data = self.train_df.copy()
        test_data = self.test_df.copy()
        
        # object나 string 타입의 컬럼만 선택
        categorical_columns = [col for col in train_data.columns 
                             if train_data[col].dtype == 'object' or 
                             train_data[col].dtype == 'string' or
                             pd.api.types.is_categorical_dtype(train_data[col])]
        
        # AutoMLPipelineFeatureGenerator 적용
        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator.fit_transform(X=train_data)
        
        # 문제 타입 자동 감지
        unique_values = len(np.unique(train_data[self.target_column]))
        if unique_values == 2:
            eval_metric = 'f1'
            problem_type = 'binary'
        else:
            eval_metric = 'accuracy'
            problem_type = 'multiclass'
        
        # 실행마다 다른 seed 사용
        seed = 42 + repetition_idx
        
        # AutoGluon 실행
        predictor = ag.TabularPredictor(
            label=self.target_column,
            eval_metric=eval_metric,
            problem_type=problem_type,
            random_state=seed
        ).fit(
            train_data=train_data,
            time_limit=300
        )
        
        # 예측
        y_pred = predictor.predict(test_data)
        
        # 평가
        return self.evaluate(
            test_data[self.target_column],
            y_pred
        )

    def run_lightautoml(self, repetition_idx):
        """LightAutoML 실행"""
        print(f"\nRunning LightAutoML (Repetition {repetition_idx+1}/{self.n_repetitions})...")
        
        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        # 실행마다 다른 seed 사용
        seed = 42 + repetition_idx
        
        # task 자동 감지
        unique_values = len(np.unique(y_train))
        if unique_values == 2:
            task = Task('binary')
        else:
            task = Task('multiclass')
        
        roles = {
            'target': self.target_column,
        }
        
        # LightAutoML 실행
        automl = TabularAutoML(
            task=task,
            timeout=300,
            reader_params = {'random_state': seed},
        )
        automl.fit_predict(self.train_df, roles=roles, verbose=1)
        
        y_pred_proba = automl.predict(X_test).data  # 확률값 출력

        # 가장 확률이 높은 클래스를 예측 라벨로 변환
        y_pred = np.argmax(y_pred_proba, axis=1)

        # 평가
        return self.evaluate(y_test, y_pred)
    
    def run_mljar(self, repetition_idx):
        """MLJAR 실행"""
        print(f"\nRunning MLJAR (Repetition {repetition_idx+1}/{self.n_repetitions})...")
        
        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        # 실행마다 다른 seed 사용 및 실행마다 다른 결과 디렉토리 지정
        seed = 42 + repetition_idx
        results_path = f"mljar_results_{self.dataset_name}_rep{repetition_idx+1}"
        
        # MLJAR 실행
        automl = AutoML(
            results_path=results_path,
            mode='Compete',
            total_time_limit=300,
            random_state=seed
        )
        automl.fit(X_train, y_train)
        
        # 예측
        y_pred = automl.predict(X_test)
        
        # 평가
        return self.evaluate(y_test, y_pred)
    
    def average_metrics(self, metrics_list):
        """여러 번 실행한 지표의 평균 계산"""
        avg_metrics = {}
        for metric in metrics_list[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in metrics_list])
        
        # 표준 편차도 계산
        std_metrics = {}
        for metric in metrics_list[0].keys():
            std_metrics[metric] = np.std([m[metric] for m in metrics_list])
            
        return avg_metrics, std_metrics
    
    def save_results(self):
        """평균 및 자세한 실행 결과 저장"""
        # 자세한 결과 저장
        with open(os.path.join(self.output_dir, f"{self.dataset_name}_detailed_results.txt"), 'w') as f:
            for model, rep_metrics in self.detailed_results.items():
                f.write(f"{model.upper()} Results:\n")
                
                # 각 반복 실행 결과
                for rep_idx, metrics in enumerate(rep_metrics):
                    f.write(f"  Repetition {rep_idx+1}:\n")
                    for metric, value in metrics.items():
                        f.write(f"    {metric}: {value:.4f}\n")
                
                # 평균 및 표준편차
                f.write("  Average:\n")
                for metric, value in self.results[model].items():
                    f.write(f"    {metric}: {value:.4f}\n")
                
                f.write("  Standard Deviation:\n")
                for metric, value in self.std_results[model].items():
                    f.write(f"    {metric}: {value:.4f}\n")
                
                f.write("-" * 50 + "\n")
        
        # 요약 결과 저장
        results_df = pd.DataFrame()
        for model, metrics in self.results.items():
            model_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
            model_df['model'] = model
            results_df = pd.concat([results_df, model_df], ignore_index=True)
        
        results_df.to_csv(os.path.join(self.output_dir, f"{self.dataset_name}_summary_results.csv"), index=False)
    
    def run_all(self):
        """모든 AutoML 모델 실행 (n번 반복)"""
        # 자세한 결과 저장을 위한 구조 초기화
        self.detailed_results = {
            'h2o': [],
            'tpot': [],
            'autogluon': [],
            'lightautoml': [],
            'mljar': []
        }
        
        self.std_results = {}
        
        try:
            # 각 모델별로 n번 반복 실행
            for rep_idx in range(self.n_repetitions):
                print(f"\n=== Repetition {rep_idx+1} of {self.n_repetitions} ===")
                
                # H2O
                h2o_metrics = self.run_h2o(rep_idx)
                self.detailed_results['h2o'].append(h2o_metrics)
                
                # TPOT
                tpot_metrics = self.run_tpot(rep_idx)
                self.detailed_results['tpot'].append(tpot_metrics)
                
                # AutoGluon
                autogluon_metrics = self.run_autogluon(rep_idx)
                self.detailed_results['autogluon'].append(autogluon_metrics)
                
                # LightAutoML
                lightautoml_metrics = self.run_lightautoml(rep_idx)
                self.detailed_results['lightautoml'].append(lightautoml_metrics)
                
                # MLJAR
                mljar_metrics = self.run_mljar(rep_idx)
                self.detailed_results['mljar'].append(mljar_metrics)
            
            # 평균 계산
            for model, metrics_list in self.detailed_results.items():
                avg_metrics, std_metrics = self.average_metrics(metrics_list)
                self.results[model] = avg_metrics
                self.std_results[model] = std_metrics
                
                print(f"\n{model.upper()} Average Results after {self.n_repetitions} repetitions:")
                for metric, value in avg_metrics.items():
                    print(f"{metric}: {value:.4f} (±{std_metrics[metric]:.4f})")
            
            # 결과 저장
            self.save_results()
            
        except Exception as e:
            print(f"Error in AutoML experiment: {str(e)}")
            raise