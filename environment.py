""" 
environment.py - 식단 추천 강화학습 환경

에이전트가 음식을 고를 때마다 점수를 주고,
하루 영양소 목표롤 얼마나 달성했는지 알려주는 역할
"""

import numpy as np
import pandas as pd


class DietEnv:
    """
    식단 추천 환경 (1 에피소드 = 1주일)

    에피소드 흐름:
        reset() → 일주일 시작
        step(action) x 21번 → 7일 x 아침/점심/저녁
        done = True → 일주일 종료  
    """

    # -- 하루 영양소 목표 (성인 평균 기준) ------------------------------------
    DAILY_GOALS = {
        "calories": 14000.0,         # 2000kcal x 7
        "carbohydrates": 1750.0,     # 250g x 7
        "protein": 420.0,            # 250g x 7
        "fat": 455.0,                # 65g x7
    }

    MEALS_PER_DAY = 3  # 하루 3끼 (아침, 점심, 저녁)
    DAYS_PER_WEEK = 7  # 일주일
    TOTAL_MEALS   = MEALS_PER_DAY * DAYS_PER_WEEK # 21 스텝

    def __init__(self, csv_path: str, max_foods: int = 500):
        """
        csv_path: 다운받은 Kaggle CSV 파일 경로
        예) DietEnv("data/food_nutrition.csv")
        max_foods: action space 크기 제한(너무 크면 학습이 느려짐)
        """
        self.food_df = self._load_data(csv_path)

        # action space가 너무 크면 학습이 매우 느려지므로 제한
        if max_foods and len(self.food_df) > max_foods:
            self.food_df = self.food_df.sample(max_foods, random_state=42).reset_index(drop=True)
            print(f"[Environment] action space 제한: {max_foods}개 음식으로 랜덤 샘플링")
        self.n_foods = len(self.food_df)  #음식 개수 = action 가짓수

        # State 크기: 영양소 4개 남은 비율 + 현재 끼니 번호 1개 = 5
        self.state_size = 6
        self.action_size = self.n_foods

        # 내부 상태 변수 (reset()에서 초기화됨)
        self.remaining = {}     # 남은 영양소
        self.meal_idx = 0       # 0 ~ 20 (총 21스텝)
        self.eaten_ids = []     # 오늘 먹은 음식 ID 목록 (반복 추천 패널티용)

    # -- 데이터 로드 ------------------------------------------------------
    def _load_data(self, path:str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # 컬럼 이름 소문자 변환 후 공백 제거
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # 필요한 컬럼 후보 목록(다양한 데이터셋 이름 대응)
        col_map = {
            "name"          : ["food", "name", "food_name", "item", "description"],
            "calories"      : ["calories", "energy", "caloric_value", "caloric value", "energy_(kcal)"],
            "carbohydrates" : ["carbohydrates", "carbs", "carbohydrate", "carbohydrate_(g)"],
            "protein"       : ["protein", "proteins", "protein_(g)"],
            "fat"           : ["fat", "total_fat", "fats", "fat_(g)", "total_fat_(g)"],
        }

        rename = {}
        for target, candidates in col_map.items():
            for c in candidates:
                if c in df.columns:
                    rename[c] = target
                    break

        df = df.rename(columns=rename)

        # 필수 컬럼만 남기고 결측치 제거
        required = ["name", "calories", "carbohydrates", "protein", "fat"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV에서 다음 컬럼을 찾을 수 없습니다: {missing}\n"
                f"현재 컬럼 목록: {list(df.columns)}"
            )
        
        df = df[required].dropna().reset_index(drop=True)
        print(f"[Environment] 음식 데이터 로드 완료: {len(df)}개 음식")
        return df
    
    # -- 프로퍼티: 현재 요일(0=월 ~ 6=일) ------------------------------------
    @property
    def current_day(self) -> int:
        return self.meal_idx // self.MEALS_PER_DAY
    
    @property
    def current_meal_of_day(self) -> int:
        return self.meal_idx % self.MEALS_PER_DAY
    
    # -- 에피소드 시작 -----------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        하루를 새로 시작.
        남은 영양소를 목표치로 초기화하고 첫 번째 끼니(아침)로 돌아감.
        반환값: 초기 state (numpy 배열)
        """
        self.remaining = dict(self.DAILY_GOALS) # 목표치로 초기화
        self.meal_idx = 0
        self.eaten_ids = []
        return self._get_state()
    
    # -- 한 스텝 진행 ------------------------------------------------------
    def step(self, action: int):
        """
        에이전트가 음식(action 번호)을 선택하면:
        1. 영양소를 섭취해서 remaining을 줄임
        2. reward 계산
        3. 다음 state 반환

        반환값: (next_state, reward, done)
            next_state  : 다음 관찰 벡터
            reward      : 이번 선택으로 받은 점수
            done        : 하루(3끼)가 끝났으면 True
        """
        assert 0 <= action < self.n_foods, f"잘못된 action: {action}"

        food = self.food_df.iloc[action]

        # -- 영양소 섭취 ----------------------------------------------------
        for nutrient in ["calories", "carbohydrates", "protein", "fat"]:
            self.remaining[nutrient] -= float(food[nutrient])

        # -- Reward 계산 --------------------------------------------------
        reward = self._compute_reward(action)

        # -- 끼니 증가 & 종료 여부 확인 ---------------------------------------
        self.eaten_ids.append(action)
        self.meal_idx += 1
        done = (self.meal_idx >= self.TOTAL_MEALS)

        next_state = self._get_state()
        return next_state, reward, done
    
    # -- State 벡터 생성 --------------------------------------------------
    def _get_state(self) -> np.ndarray:
        """
        현재 상황을 숫자 6개짜리 배열로 표현.

        [남은 칼로리 비율,   남은 탄수화물 비율, 
         남은 단백질 비율,   남은 지방 비율, 
         전체 진행률 (0~1), 현재 요일(0~1, 월=0, 일=1)]

        비율이기 때문에 0-1사이 값.(1이면 아직 하나도 안 먹은 것)
        초과 섭취하면 음수가 될 수 있음.
        """
        ratios = [
            self.remaining["calories"]      / self.DAILY_GOALS["calories"],
            self.remaining["carbohydrates"] / self.DAILY_GOALS["carbohydrates"],
            self.remaining["protein"]       / self.DAILY_GOALS["protein"],
            self.remaining["fat"]           / self.DAILY_GOALS["fat"],
        ]
        progress_norm = self.meal_idx / self.TOTAL_MEALS                # 전체 진행률 (0~1)
        day_norm = self.current_day / (self.DAYS_PER_WEEK - 1 + 1e-8)   # 요일 (0~1)
        return np.array(ratios + [progress_norm, day_norm], dtype=np.float32)
    
    # -- Reward 함수 ------------------------------------------------------
    def _compute_reward(self, action: int) -> float:
        """
        [보상 설계]
        1) 영양소 달성 점수: 남은 비율이 0에 가까울수록 높은 점수
        2) 초과 섭취 페널티: 목표를 넘으면 2배 감점
        3) 당일 반복 음식 패널티
        4) 끼니별 칼로리 분배 패널티: 아침/저녁 600kcal, 점심 800kcal 기준
        5) 하루 영양소 균형 패널티: 하루 끝날 때 단백질/탄수화물 체크
        """
        reward = 0.0
        food = self.food_df.iloc[action]

        # 1) 영양소 달성 점수: 남은 비율이 0에 가까울수록(목표 달성) 높은 점수
        for nutrient, goal in self.DAILY_GOALS.items():
            remaining_ratio = self.remaining[nutrient] / goal
            # 남은 비율이 0~1 사이면 달성 중 → 양수 보상
            # 음수면 초과 섭취 → 패널티
            if remaining_ratio >= 0:
                reward += 1.0 - remaining_ratio     # 많이 달성할수록 +
            else:
                reward += remaining_ratio * 2.0     # 초과는 2배 패널티
        
        # 2) 반복 음식 패널티: 같은 음식을 또 고르면 -1점
        today_start = (self.meal_idx // self.MEALS_PER_DAY) * self.MEALS_PER_DAY
        today_eaten = self.eaten_ids[today_start:]
        if action in today_eaten:
            reward -= 2.0

        # 3) 끼니별 칼로리 분배 패널티
        # 아침(0)=600kcal, 점심(1)=800kcal, 저녁(2)=600kcal 기준
        meal_cal_goals = {0: 600.0, 1: 800.0, 2: 600.0}
        meal_of_day = self.current_meal_of_day
        food_cal = float(food["calories"])
        meal_goal = meal_cal_goals[meal_of_day]
        meal_ratio = food_cal / meal_goal
        if meal_ratio > 2.5:    # 목표의 2.5배 초과 → 한 끼에 너무 많이
            reward -= 2.0
        elif meal_ratio < 0.1:  # 목표의 10% 미만 → 너무 적게
            reward -= 1.0

        # 4) 끼니별 단백질 최소 보장 (매 끼니 20g 이상)
        food_protein = float(food["protein"])
        if food_protein < 20.0:
            reward -= 1.5
        
        # 5) 끼니별 탄수화물 최소 보장 (매 끼니 50g 이상)
        food_carbs = float(food["carbohydrates"])
        if food_carbs < 50.0:
            reward -= 1.5

        # 6) 일별 칼로리 균형 패널티 - 하루 끝날 때마다 체크
        if (self.meal_idx + 1) % self.MEALS_PER_DAY == 0:
            today_ids = self.eaten_ids[today_start::] # 오늘 먹은 음식들
            daily_calories = sum(
                float(self.food_df.iloc[fid]["calories"]) for fid in today_ids
            )
            daily_ratio = daily_calories / 2000.0 # 하루 목표 2000kcal 기준
            if daily_ratio > 1.5:   # 3000kcal 초과 → 과식 패널티
                reward -= 1.5
            elif daily_ratio < 0.5: # 1000kcal 미만 → 결식 패널티
                reward -= 1.5

        return float(reward)
    
    # -- 현재 식단 요약 출력 -------------------------------------------------
    def render(self):
        """현재까지 먹은 음식과 남은 영양소 출력"""
        day_names = ["월", "화", "수", "목", "금", "토", "일"]
        meal_names = ["아침", "점심", "저녁"]

        print(f"\n=== 주간 식단 ({self.meal_idx}/{self.TOTAL_MEALS} 끼니) ===")
        for i, fid in enumerate(self.eaten_ids):
            day = i // self.MEALS_PER_DAY
            meal = i % self.MEALS_PER_DAY
            print(f"    {day_names[day]}요일: {meal_names[meal]}: {self.food_df.iloc[fid]['name']}")
        
        print("\n   [남은 영양소 (일주일 기준)]")
        for nutrient, goal in self.DAILY_GOALS.items():
            remaining = self.remaining[nutrient]
            pct = (goal - remaining) / goal * 100
            print(f"    {nutrient:15s}: {remaining:8.1f} 남음 (달성률 {pct:.1f}%)")