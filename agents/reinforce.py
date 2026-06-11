"""
agents/reinforce.py - REINFORCE (Policy Gradient) 에이전트

Q-learning/DQN은 "각 행동의 점수(Q값)"를 직접 학습한다면, "어떤 음식을 고를 확률"을 직접 학습함.
→ 점수가 높았던 행동은 확률을 올리고, 점수가 낮았던 행동은 확률을 내림.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


# -- 정책 신경망 ----------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    state를 받아서 각 action을 선택할 확률을 출력.
    출력층에 Softmax를 써서 모든 확률의 합이 1이 되게 함.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1), # 확률로 변환
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# -- REINFORCE 에이전트 --------------------------------------------------------------
class REINFORCEAgent:

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        hidden_size: int = 128,
    ):
        self.gamma       = gamma
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[REINFORCE] 사용 디바이스: {self.device}")

        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 에피소드 동안의 기록 (한 에피소드 끝나면 한꺼번에 학습)
        self.log_probs = [] # 선택한 행동의 log 확률
        self.rewards   = [] # 각 스텝의 보상

        # REINFORCE는 epsilon X (확률 자체를 학습)
        self.epsilon = 0.0

        print(f"[REINFORCE] 신경망 파라미터 수: {sum(p.numel() for p in self.policy_net.parameters()):,}")

    # -- 행동 선택 -------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """
        신경망이 출력한 확률 분포에서 행동을 샘플링.
        예) [사과 30%, 바나나 50%, 쌀 20%] → 확률에 따라 하나 선택
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.policy_net(state_t)

        # 확률 분포에서 샘플링
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()

        # log 확률 저장 (나중에 학습에 사용)
        self.log_probs.append(dist.log_prob(action))

        return action.item()
    
    # -- 보상 기록 -------------------------------------------------------------------
    def remember_reward(self, reward: float):
        """각 스텝의 보상 기록"""
        self.rewards.append(reward)
    
    # -- 에피소드 종료 후 학습 ----------------------------------------------------------
    def learn(self):
        """
        에피소드가 끝난 후, 각 행동이 '얼마나 좋았는지' 계산.

        Return(반환 값) = 지금 받은 보상 + gamma x 미래에 받을 보상들의 합
        → 결과가 좋았던 행동은 더 자주 하도록 확률을 올림
        → 결과가 나빴던 행동은 확률을 내림
        """
        if not self.rewards:
            return
        
        # -- 각 시점의 Return 계산 (뒤에서 앞으로) ---------------------------------------
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # 정규화: 학습 안정성을 높여줌
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # -- 정책 손실 계산 -----------------------------------------------------------
        # 좋은 결과(높은 return)였던 행동의 log_prob를 최대화
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G) # 음수: 최대화 → 최소화로
        
        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 기록 초기화
        self.log_probs = []
        self.rewards   = []

        return loss.item()
    
    # -- 저장 / 불러오기 --------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"[REINFORCE] 저장 완료: {path}")

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[REINFORCE] 불러오기 완료: {path}")