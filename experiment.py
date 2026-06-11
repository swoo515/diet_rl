"""
experiment.py - 비교 실험 자동 실행 스크립트

실험 1: 알고리즘 비교        (Q-learning vs DQN vs REINFORCE)
실험 2: 학습률 비교          (0.0001 / 0.001 / 0.01)
실험 3: Reward 설계 비교    (단순형 / 다목적형 / 패널티형)

사용법: python experiment.py -- csv data/food_nutrition.csv
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # 서버 환경에서도 그래프 저장 가능하도록
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import torch

from environment import DietEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.reinforce import REINFORCEAgent
from train import run_episode

def build_agent(name, state_size, action_size, lr, gamma, epsilon=1.0,
                batch_size=64, hidden_size=128):
    if name == "q_learning":
        return QLearningAgent(
            state_size=state_size, action_size=action_size,
            lr=lr, gamma=gamma, epsilon=epsilon,
        )
    elif name == "dqn":
        return DQNAgent(
            state_size=state_size, action_size=action_size,
            lr=lr, gamma=gamma, epsilon=epsilon,
            batch_size=batch_size, hidden_size=hidden_size,
        )
    elif name == "reinforce":
        return REINFORCEAgent(
            state_size=state_size, action_size=action_size,
            lr=lr, gamma=gamma, hidden_size=hidden_size,
        )
    else:
        raise ValueError(f"알 수 없는 에이전트: {name}")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- 한 설정으로 여러 번 실행 (평균 내기 위함) ---------------------------------------------
def run_experiment(env, agent_name, episodes, lr, gamma, batch_size=64, n_runs=3):
    """
    동일한 설정으로 n_runs번 반복 실행해서 평균 reward 반환.
    여러 번의 실행으로 운이 좋거나 나쁜 경우를 평균내 결과의 신뢰도를 높이기 위함.
    """

    all_rewards = []

    for run in range(n_runs):
        seed = run
        set_seed(seed)
        print(f"   seed={seed}")
        agent = build_agent(
            agent_name, env.state_size, env.action_size, 
            lr=lr, gamma=gamma, batch_size=batch_size,)
        rewards = []

        for ep in range(1, episodes + 1):
            r = run_episode(env, agent, agent_name, train=True)
            rewards.append(r)

            if agent_name == "dqn" and ep % 10 == 0:
                agent.update_target_network()
            if hasattr(agent, "decay_epsilon"):
                agent.decay_epsilon()
        
        all_rewards.append(rewards)
        print(f"    run {run+1}/{n_runs} 완료, 최종 평균: {np.mean(rewards[-50:]):.3f}")

    return np.array(all_rewards) # shape: (n_runs, episodes)


# -- 이동 평균 ----------------------------------------------------------------------
def moving_average(arr, window=20):
    return np.convolve(arr, np.ones(window) / window, mode="valid")


# -- 그래프 그리기 -------------------------------------------------------------------
def plot_results(results_dict, title, save_path, window=20):
    """
    여러 설정의 reward 곡선을 한 그래프에 그리기.
    평균선 + 표준편차 범위(반투명) 함께 표시.
    """
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 5))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    for i, (label, rewards_matrix) in enumerate(results_dict.items()):
        mean_r = rewards_matrix.mean(axis=0)
        std_r  = rewards_matrix.std(axis=0)

        # 이동 평균으로 부드럽게
        smooth_mean = moving_average(mean_r, window)
        smooth_std  = moving_average(std_r, window)
        x = range(window - 1, len(mean_r))

        c = colors[i % len(colors)]
        plt.plot(x, smooth_mean, label=label, color=c, linewidth=2)
        plt.fill_between(x,
                         smooth_mean - smooth_std,
                         smooth_mean + smooth_std,
                         color=c, alpha=0.15)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {save_path}")


# -- 실험 1: 알고리즘 비교 ------------------------------------------------------------
def exp1_algorithm_comparison(env, episodes, save_dir, n_runs):
    print("\n[실험 1] 알고리즘 비교: Q-learning vs DQN vs REINFORCE")
    results = {}
    for agent_name in ["q_learning", "dqn", "reinforce"]:
        print(f"  → {agent_name} 학습 중...")
        results[agent_name] = run_experiment(
            env, agent_name, episodes, lr=0.001, gamma=0.95, n_runs=n_runs
        )
    plot_results(results, "알고리즘 비교", os.path.join(save_dir, "exp1_algorithm.png"))
    return results


# -- 실험 2: 학습률 비교(DQN 기준) -----------------------------------------------------
def exp2_lr_comparison(env, episodes, save_dir, n_runs):
    print("\n[실험 2] 학습률 비교 (DQN)")
    results = {}
    for lr in [0.0001, 0.001, 0.01]:
        label = f"lr={lr}"
        print(f"  → {label} 학습 중...")
        results[label] = run_experiment(env, "dqn", episodes, lr=lr, gamma=0.95, n_runs=n_runs)
    plot_results(results, "학습률(lr) 비교 - DQN", os.path.join(save_dir, "exp2_lr.png"))
    return results
    

# -- 실험 3: 할인율 비교 (DQN 기준) ----------------------------------------------------
def exp3_gamma_comparison(env, episodes, save_dir, n_runs):
    print("\n[실험 3] 할인율(gamma) 비교(DQN)")
    results = {}
    for gamma in [0.9, 0.95, 0.99]:
        label = f"gamma={gamma}"
        print(f"  → {label} 학습 중...")
        results[label] = run_experiment(env, "dqn", episodes, lr=0.001, gamma=gamma, n_runs=n_runs)
        plot_results(results, "할인율(gamma) 비교 - DQN", os.path.join(save_dir, "exp3_gamma.png"))
    return results


# -- 최종 성능 요약표 출력 --------------------------------------------------------------
def print_summary(all_results):
    print("\n" + "="*55)
    print(f"{'설정':<25} {'최종 평균 reward':>15} {'표준편차':>10}")
    print("="*55)
    for exp_name, results in all_results.items():
        for label, rewards_matrix in results.items():
            final_mean = rewards_matrix[:, -50:].mean()
            final_std  = rewards_matrix[:, -50:].std()
            print(f"  [{exp_name}] {label:<20} {final_mean:>10.3f}  ±{final_std:.3f}")
    print("="*55)


# -- 메인 ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         type=str, default="data/food_nutrition.csv")
    p.add_argument("--episodes",    type=int, default=1000)
    p.add_argument("--n_runs",      type=int, default=5,                        help="각 설정을 몇 번 반복 실행할지 (많을수록 신뢰도 ↑)")
    p.add_argument("--save_dir",    type=str, default="results")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = DietEnv(args.csv, max_foods=500)

    all_results = {}
    all_results["exp1"] = exp1_algorithm_comparison(
        env, args.episodes, args.save_dir, args.n_runs)
    all_results["exp2"] = exp2_lr_comparison(
        env, args.episodes, args.save_dir, args.n_runs)
    all_results["exp3"] = exp3_gamma_comparison(
        env, args.episodes, args.save_dir, args.n_runs)
    
    print_summary(all_results)

    # JSON 저장
    summary = {}
    for exp_name, results in all_results.items():
        summary[exp_name] = {
            label: {
                "mean": float(r[:, -50:].mean()),
                "std" : float(r[:, -50:].std()),
            }
            for label, r in results.items()
        }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {args.save_dir}/summary.json")