"""
evaluate.py - 최종 모델 평가 스크립트

[역할]
1. 학습된 DQN 모델로 실제 일주일 식단 추천 출력
2. Random Baseline vs DQN 성능 비교
3. 결과를 텍스트 + 그래프로 저장
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import DietEnv
from agents.dqn import DQNAgent


# -- 설정 ----------------------------------------------------------------------
CSV_PATH = "data/food_nutrition.csv"
MODEL_PATH = "results/dqn_model.pt"
SAVE_DIR = "results"
MAX_FOODS = 500
N_EVAL = 100 # 평가 에피소드 수


# -- 1) Random Baseline 에이전트 ------------------------------------------------
class RandomAgent:
    """비교용 랜덤 에이전트 - 항상 무작위로 음식 선택"""
    def __init__(self, action_size):
        self.action_size = action_size
    
    def act(self, state):
        return np.random.randint(self.action_size)
    

# -- 2) 한 에피소드 실행 ----------------------------------------------------------
def run_episode(env, agent):
    state = env.reset()
    total_reward = 0.0
    while True:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


# -- 3) 성능 비교 (DQN vs Random) -----------------------------------------------
def evaluate_comparison(env, dqn_agent, n_eval=N_EVAL):
    print(f"\n[비교 평가] DQN vs Random - {n_eval}번 실행 중...")

    dqn_rewards    = [run_episode(env, dqn_agent) for _ in range(n_eval)]
    random_rewards = [run_episode(env, RandomAgent(env.action_size)) for _ in range(n_eval)]

    result = {
        "dqn"   : {"mean": float(np.mean(dqn_rewards)),    "std": float(np.std(dqn_rewards))},
        "random": {"mean": float(np.mean(random_rewards)), "std": float(np.std(random_rewards))},
    }

    print(f"  DQN    평균 reward: {result['dqn']['mean']:.3f} ±{result['dqn']['std']:.3f}")
    print(f"  Random 평균 reward: {result['random']['mean']:.3f} ±{result['random']['std']:.3f}")
    print(f"  DQN 개선율: +{(result['dqn']['mean'] - result['random']['mean']) / abs(result['random']['mean']) * 100:.1f}%")

    # 박스플롯 저장
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([dqn_rewards, random_rewards], labels=["DQN (학습됨)", "Random (무작위)"],
               patch_artist=True,
               boxprops=dict(facecolor="#EEF5F3"),
               medianprops=dict(color="#4A7C6F", linewidth=2))
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("DQN vs Random Baseline 성능 비교", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "baseline_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {path}")

    return result, dqn_rewards, random_rewards


# -- 4) 일주일 식단 추천 출력
def recommend_weekly_diet(env, agent):
    """학습된 에이전트로 일주일 식단을 추천하고 출력"""
    print("\n[식단 추천] 학습된 DQN으로 일주일 식단 생성 중...")

    state = env.reset()
    day_names = ["월", "화", "수", "목", "금", "토", "일"]
    meal_names = ["아침", "점심", "저녁"]

    weekly_diet = {day: {} for day in day_names}
    weekly_nutrition = {day: {"calories": 0, "carbohydrates": 0, "protein": 0, "fat": 0}
                        for day in day_names}
    
    step = 0
    while True:
        action = agent.act(state)
        state, reward, done = env.step(action)

        day  = step // env.MEALS_PER_DAY
        meal = step % env.MEALS_PER_DAY
        food = env.food_df.iloc[action]

        weekly_diet[day_names[day]][meal_names[meal]] = food["name"]
        for nutrient in ["calories", "carbohydrates", "protein", "fat"]:
            weekly_nutrition[day_names[day]][nutrient] += float(food[nutrient])

        step += 1
        if done:
            break

    # 출력
    print("\n" + "="*60)
    print("  📅 일주일 추천 식단")
    print("="*60)
    for day in day_names:
        print(f"\n  [{day}요일]")
        for meal in meal_names:
            food_name = weekly_diet[day].get(meal, "-")
            print(f"   {meal}: {food_name}")
        n = weekly_nutrition[day]
        print(f"    → 칼로리 {n['calories']:.0f}kcal | "
                f"탄수화물 {n['carbohydrates']:.1f}g | "
                f"단백질 {n['protein']:.1f}g | "
                f"지방 {n['fat']:.1f}g")
        
    # 주간 총합 및 달성률
    print("\n" + "="*60)
    print("  📊 주간 영양소 달성률")
    print("="*60)
    totals = {"calories": 0, "carbohydrates": 0, "protein": 0, "fat": 0}
    for day in day_names:
        for nutrient in totals:
            totals[nutrient] += weekly_nutrition[day][nutrient]

    goal_names = {"calories": "칼로리", "carbohydrates": "탄수화물",
                    "protein": "단백질", "fat": "지방"}
    for nutrient, goal in env.DAILY_GOALS.items():
        pct = totals[nutrient] / goal * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {goal_names[nutrient]:6s} [{bar}] {pct:.1f}% "
                f"({totals[nutrient]:.0f} / {goal:.0f})")
        
    # JSON 저장
    output = {"weekly_diet": weekly_diet, "weekly_nutrition": weekly_nutrition, "totals": totals}
    path = os.path.join(SAVE_DIR, "weekly_diet.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  식단 저장: {path}")

    return weekly_diet, weekly_nutrition
    

# -- 메인 
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 환경 생성
    env = DietEnv(CSV_PATH, max_foods=MAX_FOODS)

    # DQN 모델 로드
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        epsilon=0.0,    # 평가 시에는 탐험 없이 최선만 선택
    )
    agent.load(MODEL_PATH)
    print(f"[모델 로드] {MODEL_PATH}")

    # 1) 일주일 식단 추천
    weekly_diet, weekly_nutrition = recommend_weekly_diet(env, agent)

    # 2) DQN vs Random 비교
    result, dqn_rewards, random_rewards = evaluate_comparison(env, agent)

    # 요약 저장
    summary = {
        "model": MODEL_PATH,
        "max_foods": MAX_FOODS,
        "n_eval": N_EVAL,
        "comparison": result,
    }
    with open(os.path.join(SAVE_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n ✅ 평가 완료!")
    print(f"  results/weekly_diet.json        - 추천 식단")
    print(f"  results/baseline_comparison.png - 비교 그래프")
    print(f"  results/evaluation_summary.json - 수치 요약")