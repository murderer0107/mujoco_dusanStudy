# mujoco_dusanStudy

MuJoCo 기반 시뮬레이션, 제어, 강화학습 실험을 정리하는 저장소입니다.

이 저장소에는 여러 실험 코드와 예제가 함께 들어 있으며, 최근에는 pure MuJoCo + Gymnasium 기반 Cartpole swing-up 환경과 PPO 학습 흐름을 추가했습니다.


## Projects

### Cartpole

Pure MuJoCo + Gymnasium 기반 Cartpole swing-up 실험입니다.

주요 내용:

- 아래로 처진 막대를 카트 반동으로 위로 세우는 swing-up 환경
- PPO 기반 학습 및 실행 스크립트
- 외란 버전 및 강한 외란 시연 모드
- TensorBoard 로그 확인 가능

문서:

- [Quickstart](./code/Cartpole/QUICKSTART_pure_mujoco_cartpole.md)
- [Detailed README](./code/Cartpole/README_pure_mujoco_cartpole.md)


### 2DOF

2자유도 관련 실험 코드와 시각화 결과가 포함되어 있습니다.


## Repository Structure

```text
code/
  Cartpole/
    env/
    train/
    play/
    README_pure_mujoco_cartpole.md
    QUICKSTART_pure_mujoco_cartpole.md
  2dof/
bin/
include/
lib/
model/
sample/
simulate/
```


## Quick Start

### Cartpole base training

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py
```


### Cartpole base play

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py
```


### Cartpole disturbance training

```powershell
python C:\mujoco\code\Cartpole\train\train_pure_mujoco_cartpole_swingup.py --disturbance
```


### Cartpole strong-disturbance demo

```powershell
python C:\mujoco\code\Cartpole\play\play_pure_mujoco_cartpole_swingup.py --disturbance --strong-disturbance
```


## Environment

권장 Python 패키지:

- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `tensorboard`


## Notes

- 자세한 설명과 실험 배경은 각 프로젝트 폴더의 문서를 참고하면 됩니다.
- Cartpole 관련 상세 설명은 `code/Cartpole` 아래 문서에 정리되어 있습니다.
