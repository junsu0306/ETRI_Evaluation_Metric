# 구버전 실험 감사 — ResNet-18 test 이상치와 MobileNetV3 실험 현황

작성일: 2026-09-07
대상: 2026-08-31 아카이브된 구버전(legacy) 코드와 `runs/` 산출물

## 요약

1. 구버전 ResNet-18 baseline의 test Recall(R@1 0.7399)은 **ResNet-18을 측정한 값이 아니다.**
   구버전 test 모드가 로드한 checkpoint를 버리고 리포 루트의 `reduced_model.pth`
   (MobileNetV2)를 강제로 평가했다. 신규 파이프라인의 0.8078이 정상값이다.
2. 같은 표의 val 값(R@1 0.7805)은 학습 루프가 기록한 것이라 신뢰할 수 있다.
   val은 맞고 test만 틀린 비대칭이 이 결함의 직접적인 증거다.
3. MobileNetV3 실험은 존재하지만 **미완이며 문서 서술과 다르다.** run 디렉터리 32개 중
   실제 학습된 것은 5개고, KD를 적용하지 않았으며, test split은 한 번도 측정하지 않았다.

---

## 1. ResNet-18 test Recall이 낮았던 원인

### 1.1 관측된 불일치

| Split | 구버전 기록 | 신규 실행 | 차이 |
|---|---:|---:|---:|
| val R@1 | 0.7805 | 0.7896 | +0.91%p |
| val R@5 | 0.9172 | 0.9114 | -0.58%p |
| val R@10 | 0.9451 | 0.9426 | -0.25%p |
| val R@20 | 0.9670 | 0.9661 | -0.09%p |
| test R@1 | 0.7399 | 0.8078 | **+6.79%p** |
| test R@5 | 0.8750 | 0.9024 | +2.74%p |
| test R@10 | 0.9117 | 0.9300 | +1.83%p |
| test R@20 | 0.9420 | 0.9520 | +1.00%p |

val은 모든 rank에서 사실상 동일한데 test만 크게 벌어진다. 두 모델의 표현력이
같다면 test에서만 7%p 가까이 벌어질 수 없다. 측정 경로를 의심해야 하는 형태다.

Pittsburgh30k에서는 통상 test R@1이 val R@1보다 같거나 높다. 구버전 ResNet-18은
test가 val보다 4.06%p **낮아** 비정상이고, 신규는 1.82%p 높아 정상 패턴이다.

### 1.2 원인 — test 모드가 checkpoint를 버리고 다른 모델을 읽는다

`trash/legacy/root_scripts/main_variant.py:591-595`

```python
if opt.mode.lower() == 'test':
    print('===> Running evaluation step')

    model = torch.load("reduced_model.pth")
    model = model.to(device)
```

바로 위 `main_variant.py:570-586`에서 `--resume`과 `--ckpt=best`로
`model_best.pth.tar`를 로드하고 `model.load_state_dict()`까지 수행한다.
그런데 test 모드에 진입하는 순간 리포 루트의 `reduced_model.pth`로 모델 객체
전체를 덮어쓴다. `--arch`, `--resume`, `--ckpt` 인자가 모두 무시된다.

동일한 결함이 `main_for_KD.py:368`에도 있다.
`main.py`와 `main_for_teacher.py`에는 없다.

### 1.3 `reduced_model.pth`의 실제 내용

pickle 구조를 직접 열어 확인한 결과다.

```
torchvision.models.mobilenetv2 InvertedResidual   ← MobileNetV2
torch.nn.modules.activation ReLU6
torchvision.ops.misc ConvNormActivation
netvlad NetVLAD

InvertedResidual 등장 : 1
BasicBlock 등장       : 0        ← ResNet 계열 흔적 없음
```

파일 크기 7,455,577 bytes는 MobileNetV2 reduced 모델(학습 가능 파라미터
462,560개)의 규모와 일치한다. 이 파일은 `reducing.ipynb`가
`torch.save(reduction_model, "reduced_model.pth")`로 매번 덮어쓰므로,
측정 시점마다 내용이 달랐을 수 있다.

### 1.4 val은 맞고 test만 틀린 이유

두 숫자의 출처가 다르다.

| 값 | 측정 경로 | 결함 영향 |
|---|---|---|
| val 0.7805 | 학습 루프가 매 epoch `test(whole_test_set)` 호출, tfevents와 checkpoint payload에 기록 | 없음 |
| test 0.7399 | 학습 종료 후 `--mode=test`를 **별도 실행** | 직접 영향 |

구버전 checkpoint에서 추출한 메타데이터가 val 쪽이 진짜임을 확인해 준다.

```
runs/May06_07-30-02_resnet-18_netvlad/checkpoints/model_best.pth.tar
  epoch      : 18
  recalls    : {1: 0.7804942, 5: 0.9171924, 10: 0.9450578, 20: 0.9670084}
  best_score : 0.9171924
```

`PROGRESS_LEGACY.md`의 ResNet-18 val 행과 소수점까지 일치한다.
즉 **val은 ResNet-18을, test는 MobileNetV2를 측정한 값**이다.

### 1.5 방증

- 0.7399는 구버전 표의 MobileNetV2 계열 test 값(0.7061 / 0.7193 / 0.7199)과 같은 대역에 있다.
- 신규 설계 원칙에 이 결함이 이미 명시돼 있다.
  `ARCHITECTURE.md:31` — "test 시 임의의 `reduced_model.pth`를 강제로 읽지 않고
  명시된 checkpoint만 사용한다."

### 1.6 확인하지 못한 부분

0.7399를 **어느 스크립트로** 측정했는지는 실행 로그가 남아 있지 않아 단정할 수 없다.
`main.py`나 `main_for_teacher.py`로 측정했다면 이 결함은 비껴간다.
다만 `PROGRESS_LEGACY.md`가 학습 스크립트로 `main_variant.py`를 명시하고
`temp.sh`가 정확히 `main_variant.py --mode=test --split=test` 형태를 사용하므로
결함 경로일 가능성이 높다.

### 1.7 영향 범위

구버전 문서의 **test 열 전체**가 같은 경로로 측정됐을 수 있다. 아래 값은
재측정 전까지 인용하지 않는다.

| 항목 | 구버전 test R@1 | 상태 |
|---|---:|---|
| ResNet-18 baseline | 0.7399 | 신규 재측정 0.8078로 대체 |
| ResNet-18 마지막 블록 pruning | 0.6504 | 미검증 |
| MobileNetV2 원본 30 epoch | 0.7061 | 미검증 |
| MobileNetV2 pruning | 0.7193 | 미검증 |
| MobileNetV2 reducing | 0.7199 | 미검증 |

val 열은 학습 루프 출처이므로 신뢰할 수 있다.

### 1.8 확정 검증 절차

구버전 checkpoint의 state_dict 키는 신규 `PlaceRecognitionModel`과 호환된다
(`encoder.*`, `pool.centroids`, `pool.conv.weight`, 총 122개 키).
따라서 신규 파이프라인에 그대로 넣어 한 번만 돌리면 확정된다.

```bash
robot-compression test --arch resnet18 --split test \
  --checkpoint <legacy>/runs/May06_07-30-02_resnet-18_netvlad/checkpoints/model_best.pth.tar
```

- 0.80 근처가 나오면 구버전 0.7399는 측정 아티팩트로 확정되고 1.7의 표를 폐기한다.
- 0.74 근처가 나오면 모델 차이가 실재하므로 원인을 다시 조사한다.

---

## 2. 부차 요인 — 학습 자체의 개선

측정 결함과 별개로 신규 학습이 실제로 더 좋다. 구버전 tfevents에서 추출한
ResNet-18 baseline의 val R@1 곡선이다.

```
e1 :0.5802  e2 :0.6671  e3 :0.7062  e4 :0.7302  e5 :0.7467
e6 :0.7524  e7 :0.7609  e8 :0.7647  e9 :0.7763  e10:0.7722
e11:0.7789  e12:0.7766  e13:0.7783  e14:0.7787  e15:0.7827
e16:0.7810  e17:0.7788  e18:0.7805  e19:0.7830  e20:0.7826
e21:0.7810  e22:0.7767  e23:0.7821  e24:0.7825  e25:0.7766
e26:0.7846  e27:0.7822  e28:0.7825  e29:0.7823
```

- 구버전은 epoch 4에서 0.7302다. 신규는 **epoch 4에서 이미 0.7896**이다.
- 구버전이 29 epoch 동안 도달한 최대값은 0.7846(e26)으로, 신규의 epoch 4 값에 못 미친다.
- 구버전 best 선택 기준인 val R@5는 e9~e29 구간이 0.9135~0.9172로 평탄하다.
  epoch 18 선택은 사실상 노이즈이며 val R@1 최적점도 아니다.

다만 이 요인의 크기는 val 기준 0.5~0.9%p 수준이라 test의 6.79%p를 설명하지 못한다.
대부분은 1장의 측정 결함이다.

---

## 3. MobileNetV3 실험 현황

### 3.1 run 인벤토리

`runs/` 아래 `*_mobilenetv3_netvlad` 디렉터리는 32개다. 이 중 실제로 학습이
진행된 것은 5개뿐이다.

| 구분 | 개수 | 특징 |
|---|---:|---|
| 즉시 종료 | 26 | `events` 40 bytes(헤더만), checkpoint 없음 |
| 부분 종료 | 1 | May26_09-06-26, `events` 918 bytes, checkpoint 없음 |
| 실학습 | 5 | checkpoint 존재 |

즉시 종료된 27개는 2026-05-26 09:06~12:43 사이에 집중돼 있다.

### 3.2 실학습 5개 결과 (val 기준)

| run | pruning | 계획 epoch | best epoch | R@1 | R@5 | R@10 | R@20 | 종료 상태 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| May26_10-04-48 | 0.2 | 30 | 1 | 0.5528 | 0.7434 | 0.8098 | 0.8704 | 1 epoch 후 중단 |
| May26_12-50-33 | 없음 | 100 | 1 | 0.5672 | 0.7466 | 0.8074 | 0.8657 | 발산 후 중단 |
| May27_04-25-26 | 없음 | 100 | 3 | 0.5814 | 0.7627 | 0.8266 | 0.8804 | 6 epoch 하락 후 중단 |
| **May27_06-58-35** | 없음 | 100 | 23 | **0.6370** | 0.8044 | 0.8588 | 0.9118 | 23 epoch에서 중단 |
| May27_16-01-05 | 0.2 | 30 | 28 | 0.6071 | 0.7973 | 0.8553 | 0.9043 | 30 epoch 완주 |

May26_12-50-33은 학습이 무너진다.

```
e1:0.5672  e2:0.5577  e3:0.5288  e4:0.4957  e5:0.4777  e6:0.4650
e7:0.4557  e8:0.4418  e9:0.4318  e10:0.4289  e11:0.4253  e12:0.4226
```

May27_04-25-26도 e3(0.5814) 이후 하락해 e6에서 0.5657로 끝난다.
안정적으로 학습된 것은 May27_06-58-35와 May27_16-01-05 두 개다.

### 3.3 Knowledge Distillation을 적용하지 않았다

`main_for_KD.py`는 `args.py`의 `get_parser()`를 사용하며, 그 파서에는
`teacher_arch`와 `reduced` 인자가 있다. 그런데 MobileNetV3 5개 run의
`flags.json`에는 두 키가 모두 **없고**, `main_variant.py` 및
`main_for_teacher.py`의 인자 집합과 정확히 일치한다. 두 스크립트 모두 KD를
수행하지 않는다.

따라서 이 실험들은 **triplet loss 단독 학습**이다.
`PROGRESS_LEGACY.md`의 "MobileNetV3: Distillation 후 최적구조 학습 진행 중"
서술은 실제 산출물과 맞지 않는다.

### 3.4 아카이브된 소스와 checkpoint가 불일치한다

| 위치 | 정의 | encoder_dim |
|---|---|---:|
| `main_variant.py:486` | `mobilenet_v3_large` | 960 |
| `main_for_teacher.py:535` | `mobilenet_v3_small` | 96 |
| `main_for_KD.py:165` | `mobilenet_v3_small` | 96 |

실제 checkpoint에서 추출한 텐서 shape는 다음과 같다.

```
pool.centroids        : (64, 96)          -> encoder_dim = 96
pool.conv.weight      : (64, 96, 1, 1)
encoder.11.block.3.1.*                    -> features[11]에서 절단
total params          : 893,841
```

**mobilenet_v3_small**이다. 5월 실행 이후 `main_variant.py`가 v3_large로
수정된 것으로 보이며, 아카이브 소스만 보고 구조를 판단하면 안 된다.

신규 `src/robot_compression/models/factory.py`는 `mobilenet_v3_small` /
`encoder_dim=96`으로 실제 실행과 일치한다.

### 3.5 압축률 대비 성능

| 모델 | trainable params | ResNet-18 대비 최적화율 | val R@1 |
|---|---:|---:|---:|
| ResNet-18 (기준 B) | 2,132,480 | — | 0.7805 |
| MobileNetV2 reduced | 462,560 | 78.31% | 0.7337 |
| MobileNetV3-small | 308,883 | **85.52%** | 0.6370 |

MobileNetV3-small의 trainable 308,883은 마지막 블록(`encoder.11.*`, 296,595)과
NetVLAD(12,288)의 합이다(`layers[:-1]` freeze 가정).

압축률만 보면 과제 목표 80%를 이미 넘는다. 문제는 Recall로, MobileNetV2보다
val R@1이 9.67%p 낮다.

pruning 민감도도 반대다.

| 모델 | pruning 전 | pruning 후 | 변화 |
|---|---:|---:|---:|
| MobileNetV2 (reducing) | 0.7340 | 0.7337 | -0.03%p |
| MobileNetV3-small (0.2) | 0.6370 | 0.6071 | **-2.99%p** |

이미 작은 모델이라 채널을 덜어낼 여유가 없다.

### 3.6 미완 상태이므로 인용에 주의

최고 성능 run인 May27_06-58-35는 100 epoch 계획 중 23에서 끊겼고, 그 시점에도
상승 추세였다.

```
e19:0.6341  e20:0.6363  e21:0.6351  e22:0.6350  e23:0.6370
```

pruning run(May27_16-01-05)은 30 epoch를 완주했으나 e24 이후 0.605~0.607에서
평탄하다.

test split은 5개 run 모두 측정하지 않았고, `PROGRESS_LEGACY.md` 결과표에도
MobileNetV3 행이 없다. 따라서 0.6370을 "MobileNetV3의 성능"으로 확정 인용하면
과소평가일 수 있다.

---

## 4. 구버전 자산 위치와 접근 방법

동일한 산출물이 세 곳에 있다. 목적에 따라 골라 쓴다.

### 4.1 구버전 GPU 서버 (원본)

모든 구버전 tfevents 155개가 단일 Docker 컨테이너에서 생성됐다.

| 항목 | 값 | 근거 |
|---|---|---|
| 컨테이너 hostname | `bce93e8ab797` | `events.out.tfevents.<ts>.bce93e8ab797` |
| 구버전 데이터셋 | `/workspace/datasets/pittsburgh/` | `pittsburgh.py:17`, `:24` |
| 구버전 run 경로 | 프로젝트 루트 기준 `./runs/` | `flags.json`의 `runsPath`, `savePath` |
| 구버전 centroid | `./data/centroids/` | `flags.json`의 `dataPath` |
| 구버전 cache | `./cache` | `flags.json`의 `cachePath` |
| 신규 파이프라인 루트 | `/workspace/ETRI_Evaluation_Metric/` | `.env.example:3-7` |

구버전 프로젝트 루트의 절대 경로와 SSH 접속 정보는 리포에 기록돼 있지 않다.
서버 접속 후 아래로 찾는다.

```bash
# 구버전 프로젝트 루트 찾기
ls -d /workspace/*/runs/May06_07-30-02_resnet-18_netvlad 2>/dev/null
find /workspace -maxdepth 3 -name "main_variant.py" 2>/dev/null

# 데이터셋이 그대로 있는지 확인
ls /workspace/datasets/pittsburgh/datasets/pitts30k_test.mat

# 컨테이너가 살아 있는지
docker ps -a --filter "id=bce93e8ab797"
```

신규 파이프라인의 산출물(`artifacts/runs/2026*`)은 **서버에만 있고 로컬에 없다.**
`docs/EXPERIMENT_SUMMARY.md`와 `docs/REPRODUCTION_GPU0_20260831.md`가 참조하는
아래 run이 여기에 해당한다.

```
/workspace/ETRI_Evaluation_Metric/artifacts/runs/20260831_164217_resnet18_netvlad
/workspace/ETRI_Evaluation_Metric/artifacts/runs/20260831_185123_mobilenetv2_netvlad
/workspace/ETRI_Evaluation_Metric/artifacts/runs/20260901_105613_mobilenetv2_netvlad_prune0p26
```

### 4.2 로컬 사본 (가장 빠름)

현재 리포 작업 트리에 구버전 산출물이 그대로 있다. `.gitignore`로 제외돼 있을
뿐 파일은 살아 있다.

| 경로 | 용량 | 제외 규칙 |
|---|---:|---|
| `runs/` | 7.3 GB | `.gitignore:6` |
| `teacher/` | 122 MB | `.gitignore:9` |
| `vgg16_netvlad_43/` | 167 MB | `.gitignore:10` |
| 루트 `*.pth` 3개 | 101 MB | `.gitignore` `*.pth` |

이 문서의 모든 수치는 이 로컬 사본에서 추출했다. 별도 준비 없이 바로 열람할 수 있다.

```bash
ls runs/May06_07-30-02_resnet-18_netvlad/checkpoints/
ls runs/May27_06-58-35_mobilenetv3_netvlad/checkpoints/
```

### 4.3 백업 git 리포

| 항목 | 값 |
|---|---|
| 경로 | `~/Projects/netvlad-git-backup` |
| commit | `c8c5b34` (`archive`) |
| 생성 | 2026-08-31 14:51:27 +0900 |
| 파일 수 | 460 (그중 mobilenetv3 관련 75) |
| 크기 | `.git/objects` 6.5 GB |
| upstream | `Nanne/pytorch-NetVlad` |

working tree가 체크아웃돼 있지 않아 디렉터리에는 `.git`만 보인다.
로컬 사본과 대조한 결과 `.pth.tar` 87개가 **완전히 일치**하므로, 4.2가 살아 있는
동안은 이 리포를 풀 필요가 없다.

특정 파일만 꺼낼 때는 체크아웃 없이 추출한다.

```bash
git -C ~/Projects/netvlad-git-backup ls-tree -r -l c8c5b34
git -C ~/Projects/netvlad-git-backup show \
  c8c5b34:runs/May27_06-58-35_mobilenetv3_netvlad/checkpoints/model_best.pth.tar \
  > /tmp/mnv3_best.pth.tar
```

전체 체크아웃은 6.5 GB가 추가로 풀리므로 권장하지 않는다.

### 4.4 torch 없이 checkpoint 메타데이터 읽기

`.pth.tar`는 zip이므로 로컬에 torch가 없어도 `data.pkl`을 직접 언피클해
`epoch`, `recalls`, `best_score`와 텐서 shape를 추출할 수 있다.
이 문서의 3.4와 3.5 수치가 그렇게 얻은 값이다. torch 클래스는 스텁으로 대체하고
`torch._utils._rebuild_tensor_v2`만 shape를 반환하도록 바꾸면 된다.

---

## 5. 후속 조치

1. **ResNet-18 test 재측정** — 1.8의 명령으로 구버전 checkpoint를 신규
   파이프라인에서 평가해 0.7399를 확정 폐기한다. 가장 우선순위가 높다.
2. **구버전 test 열 일괄 재측정** — 1.7의 5개 항목을 같은 방식으로 다시 잰다.
   특히 "ResNet-18 마지막 블록 pruning 0.6504"는 과제 논리(ResNet 직접 pruning이
   불리하다)의 근거이므로 검증이 필요하다.
3. **문서 정정** — `PROGRESS_LEGACY.md`의 MobileNetV3 관련 서술(KD 적용)을
   실제 산출물에 맞게 수정한다.
4. **MobileNetV3 재개 검토** — 압축률 85.52%는 목표 80%를 넘고 checkpoint도 남아
   있다. 다만 미수렴 상태이므로 재개하려면 May27_06-58-35에서 이어 학습하고
   test를 측정해야 한다.
5. **`src/robot_compression/data/` 커밋** — 신규 파이프라인이 import하지만
   git에 추가된 적이 없어 서버에만 존재한다. 재현성에 직접 영향을 준다.

---

## 부록 — 이 문서의 근거 파일

| 주장 | 근거 |
|---|---|
| test 모드 결함 | `trash/legacy/root_scripts/main_variant.py:591-595`, `main_for_KD.py:368` |
| 결함 없는 스크립트 | `trash/legacy/root_scripts/main.py`, `main_for_teacher.py` |
| `reduced_model.pth` 정체 | 루트 `reduced_model.pth` pickle 구조 분석 |
| 구버전 ResNet val | `runs/May06_07-30-02_resnet-18_netvlad/` checkpoint + tfevents |
| 신규 ResNet 결과 | `docs/REPRODUCTION_GPU0_20260831.md` |
| MobileNetV3 결과 | `runs/May2[67]_*_mobilenetv3_netvlad/` checkpoint + tfevents |
| KD 미적용 | 각 run `checkpoints/flags.json` 키 집합과 `args.py` 대조 |
| 설계 원칙 | `docs/ARCHITECTURE.md:31` |
