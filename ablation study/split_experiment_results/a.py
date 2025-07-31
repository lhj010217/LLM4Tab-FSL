import pandas as pd
import os
from glob import glob
from collections import defaultdict

# CSV 파일들이 있는 디렉토리
data_dir = '.'  # 현재 디렉토리 기준 (필요 시 수정)

# 모든 CSV 파일 경로 읽기
csv_files = glob(os.path.join(data_dir, '*_16shot_run*.csv'))

# 실험별로 그룹화하여 F1 평균 계산
f1_scores = defaultdict(list)

for file in csv_files:
    # 파일명에서 실험 이름 추출
    base = os.path.basename(file)
    experiment = '_'.join(base.split('_')[:2])  # 예: "bone tumor"
    
    # CSV에서 f1 점수 읽기
    df = pd.read_csv(file)
    f1 = df['f1'].iloc[0]
    
    f1_scores[experiment].append(f1)

# 실험별 평균 F1 계산
avg_f1_per_experiment = {}
print("실험별 평균 F1 score:")
for experiment, scores in f1_scores.items():
    avg_f1 = sum(scores) / len(scores)
    avg_f1_per_experiment[experiment] = avg_f1
    print(f"{experiment}: {avg_f1:.4f}")

# 전체 평균 F1 계산
overall_avg_f1 = sum(avg_f1_per_experiment.values()) / len(avg_f1_per_experiment)
print(f"\n전체 실험 평균 F1 score: {overall_avg_f1:.4f}")
