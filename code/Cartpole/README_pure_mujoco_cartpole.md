# Pure MuJoCo + Gymnasium Cartpole Swing-Up

## 개요

이 문서는 `C:\mujoco\code\Cartpole` 아래에 새로 만든 pure MuJoCo + Gymnasium 기반 cartpole swing-up 환경과 학습/실행 흐름을 정리한 문서다.

목표는 `dm_control`의 `cartpole swingup`과 최대한 비슷한 조건을 pure MuJoCo 환경에서 재현하는 것이었다.

핵심 목표:

- 막대가 아래로 처진 상태 근처에서 시작
- 카트 반동으로 막대를 위로 세우고 유지
- 기본 버전과 외란 버전을 모두 지원
- 기존 파일은 최대한 보존하고 새 파일로 분리 구현


## 새로 추가한 파일

- `env/cartpole_dm_like_swingup.xml`
  - pure MuJoCo XML 모델
  - 카트, 슬라이더, 폴 조인트, actuator 정의

- `env/cartpole_dm_like_env.py`
  - Gymnasium 환경 구현
  - 관측, 보상, reset/step, disturbance, viewer 처리 포함

- `train/train_pure_mujoco_cartpole_swingup.py`
  - PPO 학습 스크립트
  - 기본 버전과 외란 버전을 `--disturbance` 플래그로 분기

- `play/play_pure_mujoco_cartpole_swingup.py`
  - 저장된 PPO 모델 실행 스크립트
  - 기본 실행, 외란 실행, 강한 외란 시연 지원


## 기존 파일과의 관계

기존 `env/cartpole_rl_env.py`, `train/train_dm_cartpole_ppo.py`, `play/play_dm_cartpole_ppo.py` 등은 유지했다.

이번 작업은 기존 구현을 덮어쓰지 않고, pure MuJoCo 기반 새 세트를 별도 파일로 분리해서 추가하는 방식으로 진행했다.


## 환경 설계 요약

### 1. 초기 상태

reset 시 막대가 아래를 향한 상태 근처에서 시작하도록 구성했다.

- 카트 위치: `0` 근처 작은 랜덤값
- 폴 각도: `pi` 근처 작은 랜덤값
- 속도: 작은 랜덤값

즉, 막대가 아래로 늘어진 상태에서 시작해 카트 반동으로 에너지를 넣어 swing-up 해야 한다.


### 2. 관측값

관측은 다음 5개 값으로 구성했다.

- cart position
- pole angle cosine
- pole angle sine
- cart velocity
- pole angular velocity

형태:

```python
[x, cos(theta), sin(theta), x_dot, theta_dot]
```

이 구성은 `dm_control`의 cartpole 관측 방식과 유사하게 맞췄다.


### 3. 보상 구조

보상은 `dm_control`의 swing-up 감각을 따라 다음 요소의 곱으로 구성했다.

- `upright`
  - 막대가 위로 설수록 높아짐
- `centered`
  - 카트가 중앙 근처일수록 높아짐
- `small_control`
  - 너무 큰 제어 입력을 계속 쓰지 않도록 유도
- `small_velocity`
  - 각속도가 너무 커서 불안정한 상태를 줄이도록 유도

최종적으로는 대략 아래 형태다.

```python
reward = upright * centered * small_control * small_velocity
```

즉:

- 위로 잘 세우고
- 중앙 근처에서
- 과도한 힘과 과도한 각속도를 줄이며
- 안정적으로 유지하는 정책에 높은 보상을 준다


### 4. 에피소드 길이

기본 최대 에피소드 길이는 `5000` step으로 두었다.

이 값은 시각적으로 충분히 관찰 가능하도록 늘린 값이다. 초기에 `1000` 수준은 확인용으로 너무 짧아서, 실행 중 원하는 동작을 보기 전에 리셋되는 느낌이 강했다.


### 5. actuator 세기

초기에는 카트 힘이 약해 swing-up이 잘 일어나지 않았다.

따라서 XML actuator 설정에서:

- `gear="1"` 또는 `gear="10"` 수준 대신
- 최종적으로 `gear="100"`으로 조정했다

이 변경 이후 카트가 시작하자마자 충분한 반동을 만들 수 있게 되었고, 사용자가 원하는 방향으로 swing-up 동작이 명확해졌다.


## 외란 버전 설계

외란은 환경 내부에서 actuator 입력에 랜덤 force pulse를 더하는 방식으로 구현했다.

기본 외란 학습용 설정:

- 확률: `0.03`
- 크기: `-0.15 ~ 0.15`
- 지속 시간: `30 step`

의미:

- 매 step마다 일정 확률로 외란 이벤트가 시작될 수 있다
- 외란이 시작되면 일정 시간 동안 같은 방향의 힘이 추가로 들어간다
- policy가 외란을 버티고 복원하는 능력을 학습하게 된다


## 강한 외란 시연 모드

외란이 실제로 들어가는 모습이 시각적으로 더 잘 보이도록, `play` 스크립트에는 강한 시연용 외란 옵션을 따로 두었다.

`--strong-disturbance` 사용 시 설정:

- 확률: `0.05`
- 크기: `-0.30 ~ 0.30`
- 지속 시간: `100 step`

이 모드는 학습용보다 더 강하고 오래 지속되는 외란을 적용해, 카트가 한쪽으로 밀렸다가 복원하는 장면을 더 눈에 띄게 보여주기 위한 용도다.

중요:

- 강한 외란은 플레이 전용 시연용이다
- 학습 기본 외란은 상대적으로 더 안정적인 중간 강도로 유지했다


## MuJoCo viewer UI

Python에서 실행한 MuJoCo viewer 창에 좌우 UI 패널이 보이지 않았던 문제는 viewer 실행 옵션 때문이었다.

기존:

```python
show_left_ui=False
show_right_ui=False
```

수정 후:

```python
show_left_ui=True
show_right_ui=True
```

이제 Python viewer에서도 좌우 UI 패널이 보인다.

단, `bin` 폴더의 standalone simulator와는 여전히 별개의 프로그램이므로, 창 구성과 UI가 완전히 같지는 않을 수 있다.


## 실행 순서

### 1. 기본 버전 학습

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py
```

저장 모델:

- `models/ppo_pure_mujoco_cartpole_swingup`


### 2. 기본 버전 실행

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py
```


### 3. 외란 버전 학습

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py --disturbance
```

저장 모델:

- `models/ppo_pure_mujoco_cartpole_swingup_disturb`


### 4. 외란 버전 실행

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py --disturbance
```


### 5. 강한 외란 시연

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py --disturbance --strong-disturbance
```


## 로그 확인 방법

TensorBoard로 학습 로그를 확인할 수 있다.

실행:

```powershell
tensorboard --logdir C:\mujoco\code\Cartpole\logs
```

주요 로그 폴더:

- 기본 학습: `logs/tb_pure_mujoco_cartpole`
- 외란 학습: `logs/tb_pure_mujoco_cartpole_disturb`


## 자주 보는 학습 지표

### `rollout/ep_len_mean`

- 평균 에피소드 길이
- 최대값 근처면 오랫동안 무너지지 않고 버틴다는 뜻

예:

- `5000`이면 거의 매 에피소드가 최대 길이까지 유지됨


### `rollout/ep_rew_mean`

- 평균 에피소드 총보상
- 시간이 갈수록 올라가면 학습이 진행 중이라는 뜻

이 환경에서는:

- 단순히 오래 버티는 것뿐 아니라
- 얼마나 upright를 잘 유지하는지가 반영된다


### `train/approx_kl`

- 정책이 한 번 업데이트될 때 얼마나 변했는지 보는 지표
- 너무 크면 불안정할 수 있고, 너무 작으면 변화가 거의 없을 수 있음


### `train/clip_fraction`

- PPO clipping이 실제로 적용된 비율
- 업데이트가 얼마나 clipping 구간에 걸리는지 참고할 수 있음


### `train/explained_variance`

- value function이 return을 얼마나 잘 설명하는지
- `1`에 가까울수록 좋음


### `train/value_loss`

- value network 오차
- 단독 수치보다 추세가 중요함


### `train/std`

- 연속 action 정책의 표준편차
- 탐색 강도의 대략적인 지표


## 예시 로그 해석

다음과 같은 로그 예시는 학습이 꽤 잘 되고 있다는 신호다.

```text
ep_len_mean = 5e+03
ep_rew_mean = 1.4e+03
explained_variance = 0.949
approx_kl = 0.0030
```

해석:

- `ep_len_mean = 5000`
  - 에피소드가 최대 길이까지 유지되고 있음
- `ep_rew_mean = 1400`
  - 보상이 안정적으로 쌓이고 있음
- `explained_variance = 0.949`
  - value function이 꽤 잘 맞고 있음
- `approx_kl = 0.003`
  - 정책 업데이트 폭이 과격하지 않고 안정적임

즉, 이 조합은 “오래 버티고 있고, 학습도 비교적 건강하게 진행 중”이라는 뜻으로 볼 수 있다.


## dm_control과 맞춘 부분

다음 요소들은 `dm_control cartpole swingup`의 의도를 최대한 반영했다.

- 아래로 처진 상태 근처에서 시작
- 관측에 `cos(theta)`, `sin(theta)` 사용
- swing-up + balance 목적의 smooth reward 구조
- 카트 중심 유지 항 포함
- 제어 크기와 각속도를 억제하는 항 포함


## dm_control과 다른 점 또는 단순화한 부분

완전히 동일하게 복제한 것은 아니다. 다음은 의도적으로 단순화한 부분이다.

- `dm_control`의 MJCF 자산을 그대로 재사용하지 않고, 별도 pure MuJoCo XML로 재구성
- 보상 구조는 유사하게 맞췄지만 exact implementation 전체를 1:1 복제한 것은 아님
- 외란은 별도 물리 접촉이나 외부 impulse 시스템 대신 actuator 입력에 force pulse를 추가하는 방식으로 구현
- standalone simulator와 동일한 UI/동작을 목표로 하지는 않음
- 학습 관찰 편의성을 위해 최대 step 수를 더 길게 잡음


## 현재 추천 워크플로우

1. 기본 버전으로 충분히 swing-up이 되는지 확인
2. 기본 버전 로그와 플레이를 통해 upright 유지 성능 확인
3. 그 다음 외란 버전 학습
4. 외란 버전 플레이
5. 마지막으로 `--strong-disturbance`로 시연하며 복원 능력 시각적으로 확인


## 참고 메모

현재 구현은 “먼저 실제로 잘 세워지는 환경을 만든 뒤, 그 위에 robust disturbance를 얹는다”는 방향으로 조정되었다.

따라서 이후 추가 튜닝은 아래 우선순서를 추천한다.

1. 기본 swing-up 안정화
2. 외란 강도 튜닝
3. dm_control과 더 세밀한 물리 파라미터 정합
4. 필요 시 standalone simulator 연동 또는 별도 오픈 스크립트 추가
