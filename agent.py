# 강화학습 과제02
# 경영학과 2019310303 전이훈

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, Q, mode, alpha=0, gamma=0): #파라미터로 Q-table, 학습 모드, alpha(step size), gamma(discount factor)를 지정
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.alpha=alpha
        self.gamma=gamma    
        if mode== "mc_control": #사용자가 Monte Carlo방식 선택한 경우
            self.step=self.step_mc
            self.episode=[]
            self.alpha=0.005 #실험 결과 요구사항 이상의 값을 만족한 파라미터
            self.gamma=0.5  #실험 결과 요구사항 이상의 값을 만족한 파라미터
        elif mode=="q_learning":  #사용자가 Q-learning 방식 선택한 경우
            self.step=self.step_ql
            self.alpha=0.2  #실험 결과 요구사항 이상의 값을 만족한 튜닝값
            self.gamma=0.7  #실험 결과 요구사항 이상의 값을 만족한 튜닝값


    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        return np.argmax(self.Q[state]) #argmax(state)를 기준으로 업데이트
        #return np.random.choice(self.n_actions)
        #return action

    def step_mc(self, state, action, reward, next_state, done): #Monte Carlo 방식 업데이트
        if done:
            rewards = defaultdict(lambda: np.zeros(self.n_actions))
            for history in reversed(self.episode):
                state, action, reward = history
                rewards[state][action] = reward + self.gamma * rewards[state][action]
                self.Q[state][action] += self.alpha * (rewards[state][action] - self.Q[state][action])
            self.episode.clear()
        else:
            self.episode.append((state, action, reward))
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
    
    def step_ql(self, state, action, reward, next_state, done): #Q-learning 방식 업데이트
        self.Q[state][action]+=self.alpha*(reward+self.gamma*np.max(np.max(self.Q[next_state])))-self.Q[state][action]