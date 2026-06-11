"""
agents/dqn.py - DQN (Deep Q-Network) 에이전트

Q-Learning은 표(Q-table)를 사용하지만, DQN은 표 대신 '신경망(뇌)'을 사용.
신경망은 state를 입력받아서 "각 음식을 골랐을 때 기대되는 점수"를 출력.

핵심 기법:
1. Experience Replay: 경험을 메모리에 쌓아놓고 무작위로 꺼내서 학습
    → 같은 경험을 여러 번 활용하고, 학습을 안정적으로 만듦
2. Target Network:  목표값 계산용 네트워크를 별도로 유지
    → 쫓는 목표가 흔들리지 않게 해줌
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import os


# -- 신경망 구조 -----------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    state → Q값(각 action에 대한 예상 점수)을 출력하는 신경망

    구조: 입력층 → 은닉층1 → 은닉층2 → 출력층
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

# -- DQN 에이전트 ----------------------------------------------------------------------
class DQNAgent:

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10_000,      # Replay Memory 크기
        target_update_freq: int = 10,   # 몇 에피소드마다 Target Network 업데이트
        hidden_size: int = 128,
    ):
        self.state_size         = state_size
        self.action_size        = action_size
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        # GPU 사용 가능하면 GPU, 아니면 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] 사용 디바이스: {self.device}")

        # Main Network (실제 학습)
        self.q_net      = QNetwork(state_size, action_size, hidden_size).to(self.device)
        # Target Network (목표값 계산용 - 주기적으로 Main과 동기화)
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Experience Replay Memory
        # deque: 가득 차면 오래된 경험을 자동으로 버림
        self.memory        = deque(maxlen=memory_size)
        self.episode_count = 0

        print(f"[DQN] 신경 파라미터 수: {sum(p.numel() for p in self.q_net.parameters()):,}")

    # -- 경험 저장 ---------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        """(상황, 행동, 보상, 다음상황, 종료여부) 를 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    # -- 행동 선택 ---------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """epsilon-greedy로 행동 선택(Q-Learning과 동일 개념)"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax().item())
    
    # -- 신경망 학습 ------------------------------------------------------------------
    def learn(self):
        """
        메모리에서 batch_size만큼 무작위로 꺼내서 신경망 학습.
        메모리가 batch_size보다 적으면 학습하지 않음
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 무작위 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # numpy → tensor 변환
        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # 현재 Q값
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 목표 Q값
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 손실 계산 & 역전파
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    # -- Epsilon 감소 ---------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- Target Network 업데이트 -----------------------------------------------------
    def update_target_network(self):
        """Main Network의 가중치를 Target Network에 복사"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # -- 저장 / 불러오기 --------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        print(f"[DQN] 저장 완료: {path}")

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"[DQN] 불러오기 완료: {path}")