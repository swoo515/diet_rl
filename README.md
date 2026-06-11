# 🍱 식단 추천 강화학습 프로젝트

강화학습 알고리즘을 활용해 주간 식단 추천 에이전트를 학습하는 프로젝트입니다.  
에이전트는 음식 선택을 action으로 수행하며, 칼로리·탄수화물·단백질·지방 목표 달성 정도와 식단 균형을 기준으로 reward를 받습니다.

---

## 📁 파일 구조

```text
diet_rl/
├── environment.py          # 식단 추천 강화학습 환경
├── train.py                # 단일 알고리즘 학습 스크립트
├── experiment.py           # 알고리즘/하이퍼파라미터 비교 실험
├── evaluate.py             # 학습된 DQN 모델 평가 및 식단 추천
├── agents/
│   ├── q_learning.py       # Q-learning 에이전트
│   ├── dqn.py              # DQN 에이전트
│   └── reinforce.py        # REINFORCE 에이전트
├── data/
│   └── food_nutrition.csv  # Kaggle 음식 영양소 데이터
├── results/                # 학습 모델, 그래프, 요약 JSON 저장
└── docs/
    └── diet_rl_report_김선우(A74004).pdf
```

---

## ⚙️ 설치

```bash
pip install numpy pandas torch matplotlib
```

---

## 🚀 실행 방법

### 1) 데이터 준비

Kaggle에서 받은 음식 영양소 CSV 파일을 `data/` 폴더에 넣습니다.  
기본 파일 경로는 다음과 같습니다.

```text
data/food_nutrition.csv
```

파일 이름이나 위치가 다르면 실행 시 `--csv` 인자로 경로를 지정할 수 있습니다.

---

### 2) 단일 알고리즘 학습

```bash
# DQN으로 1000 에피소드 학습
python train.py --agent dqn --episodes 1000 --lr 0.001 --gamma 0.9 --seed 42

# Q-learning으로 학습
python train.py --agent q_learning --episodes 1000 --seed 42

# REINFORCE로 학습
python train.py --agent reinforce --episodes 1000 --seed 42
```

단일 학습에서는 `--seed` 인자를 통해 난수 시드를 고정할 수 있습니다.  
최종 DQN 모델은 `seed=42`로 고정하여 학습했습니다.

학습이 끝나면 `results/` 폴더에 reward 기록과 학습된 모델이 저장됩니다.

---

### 3) 비교 실험 실행

```bash
python experiment.py --episodes 1000 --n_runs 5
```

비교 실험은 다음 세 가지로 구성됩니다.

| 실험 | 비교 대상 | 고정 설정 |
|------|-----------|-----------|
| 실험 1 | Q-learning vs DQN vs REINFORCE | lr=0.001, gamma=0.95 |
| 실험 2 | lr=0.0001 / 0.001 / 0.01 | DQN, gamma=0.95 |
| 실험 3 | gamma=0.9 / 0.95 / 0.99 | DQN, lr=0.001 |

비교 실험에서는 각 설정을 `seed=0~4`로 총 5회 반복 실행했습니다.  
평가 지표는 마지막 50 episode의 평균 reward이며, 5회 반복 결과의 평균과 표준편차로 성능을 비교했습니다.

실험 결과는 `results/` 폴더에 그래프와 `summary.json` 형태로 저장됩니다.

---

### 4) 학습된 DQN 모델 평가

학습이 완료된 DQN 모델은 아래 링크에서 다운로드할 수 있습니다.

- [Download trained DQN model](./results/dqn_model.pt)

다운로드한 모델 파일은 `results/` 폴더에 위치시킨 뒤, 아래 명령어로 평가를 실행할 수 있습니다.

```bash
python evaluate.py
```

평가 스크립트는 학습된 DQN 모델을 불러와 다음 결과를 생성합니다.

```text
results/weekly_diet.json
results/baseline_comparison.png
results/evaluation_summary.json
```

---

## 🧠 환경 설계

- 1 에피소드 = 1주일 식단 추천
- 1주일 = 7일 × 3끼 = 21번의 음식 선택
- State: 남은 영양소 비율, 전체 진행률, 요일 정보
- Action: 음식 데이터셋에서 하나의 음식 선택
- Reward: 영양소 목표 달성, 초과 섭취 패널티, 반복 음식 패널티, 끼니별 균형 등을 반영

---

## 🔧 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--lr` | 0.001 | 학습률 |
| `--gamma` | 0.95 | 할인율 |
| `--episodes` | 1000 | 학습 에피소드 수 |
| `--epsilon` | 1.0 | 초기 탐험률 |
| `--batch_size` | 64 | DQN 배치 크기 |
| `--hidden_size` | 128 | 신경망 은닉층 크기 |
| `--seed` | 42 | 난수 시드 |

---

## 📊 실험 결과 요약

각 설정별 `seed=0~4` 반복 실험 결과, DQN이 가장 높은 평균 reward를 기록했습니다.  
DQN 기준 하이퍼파라미터 비교에서는 `lr=0.001`, `gamma=0.9`가 가장 좋은 성능을 보였습니다.

| 비교 항목 | 최적 설정 | 평균 Reward | 표준편차 |
|----------|-----------|------------:|---------:|
| 알고리즘 | DQN | 25.35 | 7.35 |
| 학습률 | lr=0.001 | 25.35 | 7.35 |
| 할인율 | gamma=0.9 | 28.21 | 5.35 |

최종 DQN 모델은 `lr=0.001`, `gamma=0.9`, `seed=42` 조건에서 1,000 episode 학습했습니다.

---

## 📊 결과 파일

학습 및 평가 실행 후 `results/` 폴더에 다음과 같은 파일이 생성됩니다.

| 파일 | 설명 |
|------|------|
| `dqn_rewards.json` | DQN 학습 과정의 episode별 reward 기록 |
| `dqn_model.pt` | 학습된 DQN 모델 |
| `q_learning_model.npy` | 학습된 Q-learning Q-table |
| `reinforce_model.pt` | 학습된 REINFORCE 모델 |
| `summary.json` | 비교 실험 요약 결과 |
| `weekly_diet.json` | 학습된 DQN이 생성한 주간 추천 식단 |
| `baseline_comparison.png` | DQN과 Random Baseline 성능 비교 그래프 |
| `evaluation_summary.json` | 평가 결과 요약 |

---

## 💡 CSV 컬럼 이름 문제 해결

데이터셋마다 컬럼 이름이 다를 수 있습니다. 오류가 발생하면 아래 코드로 컬럼명을 확인합니다.

```python
import pandas as pd

df = pd.read_csv("data/food_nutrition.csv")
print(df.columns.tolist())
```

`environment.py`의 `col_map`에 해당 컬럼 이름을 추가하면 됩니다.

---

## 📄 발표 자료

프로젝트 발표 자료는 아래 링크에서 확인할 수 있습니다.

- [프로젝트 발표 PDF](./docs/diet_rl_report_김선우(A74004).pdf)
