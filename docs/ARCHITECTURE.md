# 코드 구조

| 모듈 | 역할 |
|---|---|
| `src/robot_compression/cli.py` | `check`, `smoke`, `cluster`, `train`, `reduce`, `test`; checkpoint·CSV 기록 |
| `config.py` | `.env` 로드와 저장소 기준 데이터/run/cache 경로 |
| `models/factory.py` | 백본, 학습 가능 블록, NetVLAD 모델 구성 |
| `models/netvlad.py` | assignment, residual aggregation, L2 normalization |
| `data/pittsburgh.py` | split 로드, 전처리, positive/negative 선택과 triplet 구성 |
| `training/engine.py` | feature cache, 관계 KD, triplet 학습, Recall 평가 |
| `compression/mobilenet.py` | V2/V3 zero pruning, 연결 축을 함께 줄이는 구조 축소 |
| `kd_control.py` | 최종 80% 대조군의 `0.5 × Triplet` 학습 진입점 |

실행 순서는 데이터 준비 → backbone별 centroid → teacher 학습·선택 → student 학습 →
validation 최고 모델 선택 → 구조 축소 → test다. 상세 명령은 [재현 안내](REPRODUCING.md)에 있다.

Teacher와 student는 descriptor 차원이 달라도 query–candidate 검색 분포로 KD를 계산한다.
Teacher는 고정하고 student만 gradient를 받는다. 앞쪽 backbone의 BatchNorm 통계도 학습 중 고정한다.

V3는 마지막 블록 내부 expansion 채널과 연결된 BN/depthwise/SE/projection을 줄이며 출력은
160채널로 유지한다. V2는 마지막 projection 출력 채널과 NetVLAD assignment/centroid 차원까지 줄인다.

Checkpoint는 구조 metadata와 `state_dict`를 저장한다. 구조가 축소된 checkpoint는 추론용이며
optimizer/scheduler가 제거되어 학습 재개에 사용하지 않는다.
테스트는 모델 수식, KD gradient, pruning 전후 descriptor, R@1 checkpoint 선택 및 조기 종료를 검증한다.
