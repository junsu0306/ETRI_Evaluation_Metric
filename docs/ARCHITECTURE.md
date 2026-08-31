# 코드 구조와 이전 구현 대응

## 실행 흐름

1. `cluster`: CNN local descriptor 50,000개를 표본화하고 Faiss K-means로 NetVLAD centroid를 만든다.
2. `train`: 전체 train descriptor cache를 생성하고 hard-positive/hard-negative triplet을 채굴한다.
3. student의 triplet loss와 teacher/student soft-assignment KD loss를 함께 역전파한다.
4. `--pruning`이 지정된 경우 MobileNetV2 마지막 projection output을 지정 비율로 zeroing한다.
5. `reduce`: zero filter를 실제 채널에서 제거하고 NetVLAD assignment convolution과 centroid 차원도 함께 줄인다.
6. `test`: database/query descriptor를 생성하고 Faiss L2 검색으로 Recall@N을 계산한다.

## 이전 파일 대응

| 이전 코드 | 새 코드 |
|---|---|
| `main_for_KD.py`, `args.py` | `src/robot_compression/cli.py`, `config.py` |
| `trainer.py` | `src/robot_compression/training/engine.py` |
| `netvlad.py` | `src/robot_compression/models/netvlad.py` |
| `pittsburgh.py` | `src/robot_compression/data/pittsburgh.py` |
| `utils/pruning/common.py` | `src/robot_compression/compression/mobilenet.py` |
| `reducing.ipynb` | CLI의 `reduce` 모드 |

## 수정된 주요 오류

- student KD logits를 `no_grad` 밖에서 계산해 KD gradient가 student로 흐르게 했다.
- 프루닝을 `--pruning` 사용 시에만 실행하고 `--pruning-ratio` 값을 적용한다.
- test 시 임의의 `reduced_model.pth`를 강제로 읽지 않고 명시된 checkpoint만 사용한다.
- 데이터, teacher, centroid, checkpoint가 없으면 조용히 임의 초기화하지 않고 즉시 오류를 낸다.
- 물리 GPU 2와 논리 `cuda:0`의 매핑을 하나로 통일했다.
- 전체 모델 pickle 대신 구조 메타데이터와 `state_dict`를 저장한다.
