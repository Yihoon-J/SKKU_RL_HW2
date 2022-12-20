# SKKU_RL_HW2
강화학습 과제2: Taxi-v3 환경에서 Model-Free RL 방법으로 문제 해결

## Task Requirements
agent.py 코드 수정하여 다음의 성능 향상 기준 충족
* mc-control 학습 및 최종 100회 평균 reward 가 6.0 이상 출력.
* Testing after learning 실행하여 mc-control 로 업데이트 된 Q-table 로 테스트 평균 reward 가 6.0 이상 출력.
* q-learning 학습 및 최종 100회 평균 reward 가 7.0 이상 출력.
* Testing after learning 실행하여 q-learning 으로 업데이트 된 Q-table 로 테스트 평균 reward 가 7.0 이상 출력.

## File Description
**`taxi.py`** set Open AI Gym Taxi-v2/ Taxi-v3 environment

**`agent.py`** Model-free RL Code

## Results

최종 실험 결과 
Method|Avg Reward|Test Reward
---:|:----:|:----:
**MC**|6.24|6.35
**Q-learning**|8.36|7.57
