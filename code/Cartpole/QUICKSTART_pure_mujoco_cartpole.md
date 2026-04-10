# Quickstart: Pure MuJoCo Cartpole Swing-Up

## 목적

`dm_control cartpole swingup`과 비슷한 조건을 pure MuJoCo + Gymnasium 환경으로 재현한 버전이다.

핵심 목표:

- 아래로 처진 막대를 카트 반동으로 위로 세우기
- upright 상태를 가능한 오래 유지하기
- 이후 외란이 들어와도 복원 가능하게 만들기


## 핵심 파일

- 환경: `C:\mujoco\code\Cartpole\env\cartpole_dm_like_env.py`
- XML: `C:\mujoco\code\Cartpole\env\cartpole_dm_like_swingup.xml`
- 학습: `C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py`
- 실행: `C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py`
- 상세 문서: `C:\mujoco\code\Cartpole\README_pure_mujoco_cartpole.md`


## 실행 순서

### 1. 기본 버전 학습

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py
```


### 2. 기본 버전 실행

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py
```


### 3. 외란 버전 학습

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py --disturbance
```


### 4. 외란 버전 실행

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py --disturbance
```


### 5. 강한 외란 시연

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py --disturbance --strong-disturbance
```


## 저장되는 모델

- 기본 모델: `C:\mujoco\code\Cartpole\models\ppo_pure_mujoco_cartpole_swingup`
- 외란 모델: `C:\mujoco\code\Cartpole\models\ppo_pure_mujoco_cartpole_swingup_disturb`


## 현재 기본 설정

- 초기 상태: 막대가 아래를 향한 상태 근처에서 시작
- 관측: `[x, cos(theta), sin(theta), x_dot, theta_dot]`
- 최대 에피소드 길이: `5000 step`
- actuator gear: `100`


## 외란 설정

### 학습용 외란

- 확률: `0.03`
- 크기: `-0.15 ~ 0.15`
- 지속 시간: `30 step`


### 강한 시연용 외란

- 확률: `0.05`
- 크기: `-0.30 ~ 0.30`
- 지속 시간: `100 step`


## 로그 확인

TensorBoard 실행:

```powershell
tensorboard --logdir C:\mujoco\code\Cartpole\logs
```

주요 로그 폴더:

- 기본: `C:\mujoco\code\Cartpole\logs\tb_pure_mujoco_cartpole`
- 외란: `C:\mujoco\code\Cartpole\logs\tb_pure_mujoco_cartpole_disturb`


## 빠른 해석

- `rollout/ep_len_mean`
  - 높을수록 오래 버팀
- `rollout/ep_rew_mean`
  - 높을수록 swing-up과 유지가 잘 됨
- `train/explained_variance`
  - `1`에 가까울수록 value 학습이 잘 됨


## 참고

- Python MuJoCo viewer는 UI 패널이 보이도록 수정되어 있다.
- `bin`의 standalone simulator와는 별도 프로그램이라 창 구성이 완전히 같지는 않다.
- 상세 배경과 설계 설명은 `README_pure_mujoco_cartpole.md`를 참고하면 된다.
