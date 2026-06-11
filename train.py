"""
train.py - 학습 실행 스크립트

[인자 설명]
-- agent: 사용할 알고리즘 (q_learning / dqn / reinforce)
-- episode: 학습할 에피소드 수 (기본 1000)
-- csv: 데이터 파일 경로 (기본 data/food_nutrition.csv)
-- lr: 학습률 (기본 0.001)
-- gamma: 할인율 (기본 0.95)
-- batch_size: DQN 배치 크기 (기본 64)
"""

import argparse
import numpy as np
import json
import os
import random
import torch
from environment import DietEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.reinforce import REINFORCEAgent


# -- 인자 파싱 ------------------------------------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="식단 추천 강화학습 학습 스크립트")
    p.add_argument("--agent",       type=str,       default="dqn",                      choices=["q_learning", "dqn", "reinforce"])
    p.add_argument("--episodes",    type=int,       default=1000)
    p.add_argument("--seed",        type=int,       default=42)
    p.add_argument("--csv",         type=str,       default="data/food_nutrition.csv")
    p.add_argument("--lr",          type=float,     default=0.001)
    p.add_argument("--gamma",       type=float,     default=0.95)
    p.add_argument("--epsilon",     type=float,     default=1.0)
    p.add_argument("--batch_size",  type=int,       default=64)
    p.add_argument("--hidden_size", type=int,       default=128)
    p.add_argument("--save_dir",    type=str,       default="results")
    return p.parse_args()


# -- 에이전트 생성 ---------------------------------------------------------------------------------------------------------------------
def build_agent(name, state_size, action_size, args):
    if name == "q_learning":
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
        )
    elif name == "dqn":
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
        )
    elif name == "reinforce":
        return REINFORCEAgent(
            state_size=state_size,
            action_size=action_size,
            lr=args.lr,
            gamma=args.gamma,
            hidden_size=args.hidden_size,
        )


# -- seed 고정 함수 ------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- 한 에피소드 실행 -----------------------------------------------------------------------------------------------------------------
def run_episode(env, agent, agent_name: str, train: bool = True):
    """
    1 에피소드 = 하루(아침/점심/저녁 3번의 선택)

    반환값:
        total_reward : 하루 동안 받은 총 보상
    """
    state = env.reset()
    total_reward = 0.0

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward 

        if train:
            if agent_name == "q_learning":
                agent.learn(state, action, reward, next_state, done)
            
            elif agent_name == "dqn":
                agent.remember(state, action, reward, next_state, done)
                agent.learn()

            elif agent_name == "reinforce":
                agent.remember_reward(reward)
        
        state = next_state
        if done:
            break

    # REINFORCE는 에피소드 끝난 후 한꺼번에 학습        
    if train and agent_name == "reinforce":
        agent.learn()
    
    return total_reward


# --  메인 학습 루프 -----------------------------------------------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    print(f"\n{'='*50}")
    print(f"  알고리즘: {args.agent.upper()}")
    print(f"  에피소드: {args.episodes}")
    print(f"  학습률:   {args.lr}")
    print(f"  할인율:   {args.gamma}")
    print(f"  Seed:   {args.seed}")
    print(f"{'='*50}\n")

    # 환경 & 에이전트 생성
    env   = DietEnv(args.csv, max_foods=500)
    agent = build_agent(args.agent, env.state_size, env.action_size, args)

    # 결과 저장용
    reward_history  = []
    epsilon_history = []
    log_interval    = max(1, args.episodes // 20) # 5%마다 출력
    
    # -- 학습 루프 -------------------------------------------------------
    for ep in range(1, args.episodes + 1):
        total_reward = run_episode(env, agent, args.agent, train=True)
        reward_history.append(total_reward)
        epsilon_history.append(getattr(agent, "epsilon", 0.0))

        # DQN Target Network 업데이트
        if args.agent == "dqn" and ep % agent.target_update_freq == 0:
            agent.update_target_network()

        # Epsilon 감소(Q-learning, DQN)
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()

        # 진행 상황 출력
        if ep % log_interval == 0:
            recent_avg = np.mean(reward_history[-log_interval:])
            eps_str = f"  epsilon={agent.epsilon:.3f}" if hasattr(agent, "epsilon") and agent.epsilon > 0 else ""
            print(f"  Epsiode {ep:4d}/{args.episodes} | "
                  f"최근 평균 reward: {recent_avg:7.3f}{eps_str}")
    
    # -- 결과 저장 -------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    # 학습 기록 저장
    result_path = os.path.join(args.save_dir, f"{args.agent}_rewards.json")
    with open(result_path, "w") as f:
        json.dump({
            "agent"         : args.agent,
            "episodes"      : args.episodes,
            "lr"            : args.lr,
            "gamma"         : args.gamma,
            "reward_history": reward_history,
        }, f, indent=2)
    print(f"\n[결과 저장] {result_path}")

    # 모델 저장
    model_path = os.path.join(args.save_dir, f"{args.agent}_model")
    if args.agent == "q_learning":
        agent.save(model_path + ".npy")
    else:
        agent.save(model_path + ".pt")
    
    print(f"\n학습 완료! 최종 평균 reward (마지막 50 에피소드): "
          f"{np.mean(reward_history[-50:]):.3f}")
    
    return reward_history


# -- 진입점 ------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args) 