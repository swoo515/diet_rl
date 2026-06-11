import numpy as np
import os


class QLearningAgent:
    """
    Q-table 기반 에이전트.

    Q-table
        행(row) = 상황(state)
        열(col) = 선택(action)
        값      = 이 상황에서 이 행동을 하면 기대되는 점수
    """

    def __init__(
            self,
            state_size: int,
            action_size: int,
            lr: float = 0.01,               # 학습률: 새로운 경험을 얼마나 빠르게 경험할지
            gamma: float = 0.95,            # 할인율: 미래 보상을 현재 가치로 얼마나 환산할지
            epsilon: float = 1.0,           # 탐험률: 처음엔 랜덤, 점점 학습한 것 활용
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.995,
            n_bins: int = 5,               # state 각 차원을 몇 구간으로 나눌지
    ):
        self.action_size    = action_size
        self.lr             = lr
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.n_bins         = n_bins

        # state 각 차원의 범위 (-1.5 ~ 1.5 사이로 클리핑)
        self.state_bins = [
            np.linspace(-1.5, 1.5, n_bins + 1) for _ in range(state_size) 
        ]

        # Q-table 초기화 (0으로 채움)
        # 크기: (n_bins)^state_size x action_size
        table_shape  = tuple([n_bins] * state_size) + (action_size,)
        self.q_table = np.zeros(table_shape)

        print(f"[Q-Learning] Q-table 크기: {table_shape}")
        print(f"[Q-Learning] 총 파라미터 수: {self.q_table.size:,}")

    # -- State 이산화 -----------------------------------
    def _discretize(self, state: np.ndarray) -> tuple:
        """
        연속적인 숫자 state를 정수 인덱스로 변환.
        예) [0.6, 0.6, 0.67, 0.54, 0.0] → (6, 6, 7, 5, 0)
        """
        indices = []
        for i, val in enumerate(state):
            idx = np.digitize(val, self.state_bins[i]) - 1
            idx = np.clip(idx, 0, self.n_bins - 1)
            indices.append(idx)
        return tuple(indices)
    
    # -- 행동 선택 --------------------------------------
    def act(self, state: np.ndarray) -> int:
        """
        epsilon-greedy 전략:
        - epsilon 확률로 랜덤 선택 (탐혐, exploration)
        - (1 - epsilon) 확률로 Q-table에서 가장 높은 것 선택(활용, exploitation)

        처음엔 뭘 골라야 좋은지 모르니까 랜덤으로 많이 시도.
        점점 경험이 쌓이면 학습한 것을 활용
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size) # 랜덤 탐험
        
        idx = self._discretize(state)
        return int(np.argmax(self.q_table[idx]))       # 최선 선택
    
    # -- Q-table 업데이트 ------------------------------
    def learn(self, state, action, reward, next_state, done):
        """
        [벨만 방정식]
        '이번 행동의 실제 가치' = 지금 받은 점수 + 미래에 받을 점수

        Q(s, a) ← Q(s, a) + lr x [reward + gamma x max Q(s') - Q(s, a)]
                                    ↑ TD 오차(예측과 실제의 차이)
        """
        idx      = self._discretize(state)
        next_idx = self._discretize(next_state)

        current_q = self.q_table[idx][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_idx])

        # Q-table 업데이트
        self.q_table[idx][action] += self.lr * (target_q - current_q)

    # -- Epsilon 감소 ----------------------------------
    def decay_epsilon(self):
        """에피소드가 끝날 때마다 탐험률을 조금씩 줄임"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    # -- 저장 / 불러오기 ---------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)
        print(f"[Q-Learning] 저장 완료: {path}")

    def load(self, path: str):
        self.q_table = np.load(path)
        print(f"[Q-Learning] 불러오기 완료: {path}")