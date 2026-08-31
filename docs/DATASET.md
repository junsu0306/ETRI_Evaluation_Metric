# Pittsburgh 데이터셋

## 출처

- 공개 배포 디렉터리: <https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/>
- 원본 NetVLAD 데이터 명세: `netvlad_v100_datasets.tar.gz`

기본 다운로드 구성은 Pittsburgh 30k에 필요한 database image archive `000.tar`부터 `006.tar`, 실제 query image인 `queries_real.tar`, train/validation/test split을 정의하는 MATLAB 명세 파일입니다. `PITTSBURGH_FULL=1`을 지정하면 Pittsburgh 250k용 `007.tar`부터 `010.tar`도 받습니다.

## 로컬 구조

```text
data/datasets/pittsburgh/
├── images/
│   ├── 000/
│   ├── ...
│   └── 010/
├── queries_real/
└── datasets/
    ├── pitts30k_train.mat
    ├── pitts30k_val.mat
    ├── pitts30k_test.mat
    ├── pitts250k_val.mat
    └── pitts250k_test.mat
```

아카이브는 `data/downloads/pittsburgh`에 보존됩니다. 다시 실행해도 이미 내려받은 바이트와 압축 해제 marker를 재사용합니다.

## 검증

```bash
source .venv/bin/activate
source .env
robot-compression check
```

`check`는 필수 디렉터리와 `.mat` 파일을 확인한 다음 validation split의 database/query 개수를 출력합니다.
