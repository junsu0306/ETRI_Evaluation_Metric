# Pittsburgh 데이터셋

기본 다운로드 구성은 Pittsburgh30k다. 배포 원본은
<https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/>이며,
NetVLAD의 `netvlad_v100_datasets.tar.gz` split 명세를 사용한다.
공개 Git에는 이미지나 다운로드 아카이브를 포함하지 않는다.

| Split | Database | Query | 사용 |
|---|---:|---:|---|
| train | 10,000 | 7,416 | centroid, 학습, teacher descriptor cache |
| validation | 10,000 | 7,608 | checkpoint 선택과 early stopping |
| test | 10,000 | 6,816 | 선택 완료 후 최종 평가 |

```bash
./scripts/check_resources.sh
./scripts/download_pittsburgh.sh
.venv/bin/python -m robot_compression.cli check
```

다운로드 스크립트는 `000.tar`–`006.tar`, `queries_real.tar`, split 명세와 MD5 목록을 받는다.
MD5 검사 후 압축을 풀며, 완료 marker를 재사용한다. `PITTSBURGH_FULL=1`은 추가 250k 아카이브를
받는 선택 기능이고 최종 실험에는 필요하지 않다.

```text
data/datasets/pittsburgh/
├── images/000/ ... 006/
├── queries_real/
└── datasets/
    ├── pitts30k_train.mat
    ├── pitts30k_val.mat
    └── pitts30k_test.mat
```

다운로드 원본은 `data/downloads/pittsburgh/`에 보관한다. 경로를 바꿀 때는 `.env`의
`PITTSBURGH_ROOT`와 다운로드 스크립트에 전달하는 같은 이름의 환경 변수를 일치시킨다.
기본 경로 사용 시 별도 설정은 필요 없다. `check`는 디렉터리와 `.mat` 파일을 검사하고
validation DB/query 수를 출력한다. CPU smoke와 단위 테스트에는 데이터셋이 필요 없다.
