import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import h2o
from h2o.automl import H2OAutoML
from tabpfn import TabPFNClassifier
from tpot import TPOTClassifier
import autogluon.tabular as ag
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from supervised import AutoML
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUTOML_TIME_LIMITATION_SEC = 20 *60 # 30 minutes 

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


class AutoMLExperiment:
    def __init__(self, train_df, test_df, target_column, dataset_name):
        self.train_df = train_df
        self.test_df = test_df
        self.target_column = target_column
        self.dataset_name = dataset_name
        self.results = {}
        
        # 결과 저장을 위한 디렉토리 생성
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "automl_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, y_true, y_pred):

        unique_classes = len(np.unique(y_true))
        
        if unique_classes > 2:
            f1_average = 'weighted'
        else:
            f1_average = 'binary'
            
        metrics = {
            'f1': f1_score(y_true, y_pred, average=f1_average),
            'accuracy': accuracy_score(y_true, y_pred)
        }
            
        return metrics
    
    def run_h2o(self):
        print("\nRunning H2O AutoML...")
        h2o.init()
        
        train_h2o = h2o.H2OFrame(self.train_df)
        test_h2o = h2o.H2OFrame(self.test_df)
        train_h2o[self.target_column] = train_h2o[self.target_column].asfactor()

        aml = H2OAutoML(seed=42, 
                        max_runtime_secs=AUTOML_TIME_LIMITATION_SEC,
                        )
        
        aml.train(x=list(set(self.train_df.columns) - {self.target_column}),
                 y=self.target_column,
                 training_frame=train_h2o,
                 )
        
        # 예측
        pred = aml.predict(test_h2o)
        y_pred_proba = pred['predict'].as_data_frame().values
        y_pred = np.argmax(y_pred_proba, axis=1)
        # 평가
        self.results['h2o'] = self.evaluate(
            self.test_df[self.target_column],
            y_pred
        )

        print(pred)
        print(y_pred_proba)
        print(self.results['h2o'])
        
        h2o.cluster().shutdown()
    
    def run_tpot(self):
        """TPOT 실행"""
        print("\nRunning TPOT...")
        
        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        # TPOT 실행
        tpot = TPOTClassifier(random_state=42, 
                              scoring = 'f1_weighted',
                              max_time_mins=AUTOML_TIME_LIMITATION_SEC / 60, 
                              n_jobs=-1,)
        tpot.fit(X_train, y_train)
        
        # 예측
        y_pred = tpot.predict(X_test)
        
        # 평가
        self.results['tpot'] = self.evaluate(y_test, y_pred)
        print(self.results['tpot'])
    
    def run_autogluon(self):
        """AutoGluon 실행"""
        print("\nRunning AutoGluon...")
        
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
        
        # AutoGluon 실행
        predictor = ag.TabularPredictor(
            label=self.target_column,
            eval_metric=eval_metric,
            problem_type=problem_type
        ).fit(
            train_data=train_data,
            time_limit=AUTOML_TIME_LIMITATION_SEC
        )
        
        # 예측
        y_pred = predictor.predict(test_data)
        
        # 평가
        self.results['autogluon'] = self.evaluate(
            test_data[self.target_column],
            y_pred
        )
    
        print(self.results['autogluon'])


    def run_lightautoml(self):
        """LightAutoML 실행"""
        print("\nRunning LightAutoML...")
        
        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
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
            timeout=AUTOML_TIME_LIMITATION_SEC,
            reader_params = {'random_state': 42},
            tuning_params={
                        "models": {"LGBM": {"class_weight": "balanced"}}
                    }
        )
        automl.fit_predict(self.train_df, roles=roles, verbose=1)
        
        y_pred_proba = automl.predict(X_test).data  
        print(y_pred_proba)
        y_pred = np.argmax(y_pred_proba, axis=1)

        self.results['lightautoml'] = self.evaluate(y_test, y_pred)
        print(self.results['lightautoml'])
    
    def run_mljar(self):
        """MLJAR 실행"""
        print("\nRunning MLJAR...")
        
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        # MLJAR 실행
        automl = AutoML(
            results_path=f"mljar_results_{self.dataset_name}",
            mode='Compete',
            total_time_limit = AUTOML_TIME_LIMITATION_SEC,
        )
        automl.fit(X_train, y_train)
        
        y_pred = automl.predict(X_test)
        
        self.results['mljar'] = self.evaluate(y_test, y_pred)

        print(self.results['mljar'])
    
    def run_tabpfn(self):
        """TABpfn 실행"""
        print("\nRunning TABpfn...")

        # 데이터 준비
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]
        
        tabpfn = TabPFNClassifier(device="cpu")  
        tabpfn.fit(X_train.values, y_train.values)

        y_pred = tabpfn.predict(X_test.values)

        self.results['tabpfn'] = self.evaluate(y_test, y_pred)
        print(self.results['tabpfn'])
    
    def save_results(self, run):
        run_dir = os.path.join(self.output_dir, f"8shot_run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, f"{self.dataset_name}_results.txt"), 'w') as f:
            for model, metrics in self.results.items():
                f.write(f"{model.upper()} Results:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("-" * 50 + "\n")
    
    def run_all(self, run):
        """모든 AutoML 모델 실행"""
        try:
            self.run_h2o()
            self.run_tpot()
            self.run_autogluon()
            #self.run_lightautoml()
            self.run_mljar()
            self.run_tabpfn()
            self.save_results(run)
        except Exception as e:
            print(f"Error in AutoML experiment: {str(e)}")
            raise


