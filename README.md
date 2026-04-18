# GCP 기반 실용형 광고 ML 파이프라인 (CTR Prediction)

GCP(Google Cloud Platform) 인프라와 최신 추천/광고 모델 아키텍처를 결합하여 만든 엔드투엔드(End-to-End) 머신러닝 파이프라인 포트폴리오입니다. 클릭률(CTR, Click-Through Rate) 예측을 위한 데이터 적재, 특성 공학, 파이프라인 오케스트레이션, 그리고 서빙까지 모든 단계를 포함합니다.

## 🎯 프로젝트 핵심 목표

1. **대용량 데이터 파이프라인 구축**: 로컬 OOM(Out of Memory) 한계를 극복하는 BigQuery Chunking 및 TensorFlow 데이터 스트리밍.
2. **Kubeflow 파이프라인 자동화**: 관리형 서버리스 환경인 GCP Vertex AI Pipelines를 통한 전처리-학습-평가 파이프라인 CI/CD 기반 마련.
3. **최신 모델 아키텍처 적용**: 기본형 단순 DNN에서 벗어나, 상호작용 피처를 스스로 학습하는 **DCN V2**와 **AutoInt(Transformer)** 비교 평가.
4. **온라인 서빙 (TF Serving & Vertex Endpoint)**: 훈련된 `SavedModel`을 파이프라인 상에서 즉시 Endpoint로 배포하여 안정적인 REST API 추론 환경 완성.

---

## 🏗️ 아키텍처 및 구현 기술

### 1. 인프라스트럭처 (IaC)
- **리소스**: GCS Bucket, BigQuery Dataset, Artifact Registry
- **이미지 관리**: Artifact Registry를 활용한 커스텀 베이스 이미지 관리 (의존성 설치 속도 및 버전 고정)

### 2. 데이터 & 전처리 병목 해소 (Data Scalability)
- BigQuery 데이터 조회 시 `to_dataframe()` 대신 Chunk 기반 DataFrame 로딩 및 Append 방식을 구현하여 OOM을 원천 차단했습니다.
- 모델 훈련 시 `pd.read_csv()`를 배제하고 `tf.data.experimental.make_csv_dataset`을 활용, 디스크에서 GPU/CPU로 직접 Streaming 및 동적 배칭(Batching)을 수행합니다.

### 3. 모델 아키텍처 진화 (Feature Engineering ➡️ Self-Learning)
초기 프로젝트 세팅에서는 데이터 탐색(EDA)을 거쳐 `banner_x_device`와 같은 설명 가능한 수동 교차 변수(Handcrafted Cross Feature)를 SQL 단위에서 직접 만들었습니다.

하지만 **현재 아키텍처에서는 이러한 수동 피처들을 모두 제거**했습니다. 대신, 딥러닝 스스로 피처 간 규칙을 찾는 고도화된 모델을 도입하여 백엔드 데이터 서빙 부담을 덜고 확장성을 높였습니다. 
- **DCN V2 (Deep & Cross Network V2)**: 현재 파이프라인의 **베이스라인(Baseline) 모델**. 임베딩을 선형적으로 교차 학습(Cross Network)함과 동시에 DNN 계층에 통과시켜 양쪽 특징을 종합하는 효율적인 클릭률 예측 구조입니다.
- **AutoInt (Automatic Feature Interaction)**: 비교용으로 도입된 **Transformer 기반 아키텍처**. 임베딩을 고정 차원으로 투영하여 MultiHeadAttention을 적용해 피처 간 다이내믹 집중도(Self-Attention)를 모델링합니다.

---

## 🚀 실행 가이드 (CLI Commands)

프로젝트 루트 디렉토리에서 단계별로 다음 파이썬 파일 및 CLI 명령어를 수행하여 파이프라인을 구축할 수 있습니다.

### Step 0: 환경 설정 및 인프라 프로비저닝
데이터는 [Kaggle Avazu CTR Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction/data) 데이터를 사용합니다.
```bash
# 파이썬 의존성 설치
uv sync

# 인프라 생성
cd terraform
terraform init
terraform apply -var="bucket_name=<YOUR_BUCKET_NAME>"
cd ..
```

### Step 1: 원본 데이터 업로드 및 BigQuery 적재
GCS로 다운로드한 훈련 데이터를 복사하고, BQ Raw 테이블을 생성합니다.
```bash
# 1. GCS로 Raw 데이터 업로드
gsutil cp train.csv gs://<YOUR_BUCKET_NAME>/avazu/train.csv

# 2. BigQuery 원본 테이블 적재
uv run python src/bq_load.py \
    --project <YOUR_PROJECT_ID> \
    --dataset <YOUR_DATASET_ID> \
    --gcs_uri gs://<YOUR_BUCKET_NAME>/avazu/train.csv
```

### Step 2: Feature View 생성
단순한 날짜 파싱 및 결측치 처리를 수행하는 뷰테이블을 생성합니다.
```bash
# SQL 파일 내 환경변수를 치환하며 BQ View 생성
uv run python src/run_query.py \
    --sql_file sql/create_feature_view.sql \
    --project <YOUR_PROJECT_ID> \
    --dataset <YOUR_DATASET_ID>
```

### Step 3: (선택) 로컬 파이프라인 테스트
오케스트레이션 배포 전, 로컬 혹은 VM 상에서 모델 아키텍처 단위 테스트를 진행할 수 있습니다.
```bash
# 1. Chunk 기반 전처리 및 CSV 추출
uv run python src/preprocess.py \
    --project <YOUR_PROJECT_ID> \
    --dataset <YOUR_DATASET_ID> \
    --out_dir /tmp/ctr_data \
    --sample_rows 2000000

# 2-A. DCN V2 베이스라인 훈련
uv run python src/train.py \
    --data_dir /tmp/ctr_data \
    --model_dir /tmp/ctr_model_dcn \
    --model_type dcn_v2

# 2-B. AutoInt(Transformer) 아키텍처 훈련
uv run python src/train.py \
    --data_dir /tmp/ctr_data \
    --model_dir /tmp/ctr_model_autoint \
    --model_type autoint
```

**[선택] 로컬 TF Serving 추론 테스트**
로컬에서 훈련을 잘 마쳤다면, 도커 컨테이너를 하나 띄워 곧바로 REST API가 동작하는지 검증할 수 있습니다.
```bash
# 1. 학습 결과 폴더 경로를 넘겨주어 로컬 포트(8501)에 TF Serving 실행
uv run python run_serving.py --model_uri /tmp/ctr_model_dcn

# 2. (다른 터미널 창을 열고) 로컬 서빙 전용 테스트 스크립트 발송
uv run python src/serving_request.py
```

### Step 4: (중요) 커스텀 베이스 이미지 빌드 및 푸시
파이프라인 컴포넌트 실행 속도 최적화 및 실행 환경의 일관성을 위해 모든 의존성이 포함된 커스텀 이미지를 사용합니다. Terraform으로 생성된 Artifact Registry에 이미지를 푸시해야 합니다.

```bash
# 1. .env.example 참고하여 .env 파일 생성 및 값 설정

# 2. 이미지 빌드 및 Artifact Registry 푸시 (스크립트가 .env를 자동으로 읽습니다)
bash build_and_push.sh
```

### Step 5: Kubeflow 파이프라인 컴파일 및 실행
작성된 KFP 스크립트를 컴파일하여 Vertex AI Pipelines 환경에서 실행시킵니다. 이제 `eval-op`와 `deploy_op`의 결과가 GCS에 Artifact로 저장되며, Vertex AI Experiments 탭에서 지표를 즉시 확인할 수 있습니다.

```bash
# 1. KFP yaml 파일(ctr_pipeline.yaml) 생성
export PYTHONPATH=$PYTHONPATH:.
uv run python pipelines/compile.py

# 2. Vertex AI 하위 파이프라인으로 자동 제출(Submit)
# .env 파일에 값이 설정되어 있다면 인자를 생략할 수 있습니다.
uv run python pipelines/submit_pipeline.py
```
> **💡 팁**: `.env` 파일에 `GCP_PROJECT_ID`, `PIPELINE_ROOT` 등이 설정되어 있으면 별도의 인자 없이 실행 가능합니다.
> 제출 완료 후 터미널에 출력되는 URL을 클릭하여 GCP Vertex AI 파이프라인 콘솔에서 실시간 진행 상황을 모니터링할 수 있습니다.

### Step 6: Vertex AI Endpoint를 통한 온라인 추론 (Serving 테스트)
파이프라인의 최종 단계(`deploy-op`)가 성공적으로 종료되면 GCP 콘솔의 **Vertex AI > 엔드포인트**에 자동으로 모델이 배포되어 REST API를 제공합니다.
GCP 전용 추론 스크립트를 통해 원격지에서 모델에 데이터를 보내 실시간 클릭률(CTR) 예측값을 테스트해 볼 수 있습니다.

```bash
# 콘솔에서 생성된 엔드포인트 ID(숫자)를 확인 후 입력
uv run python src/vertex_request.py
```

> **⚠️ 주의 (과금 주의)**  
> 추론 전용 머신 인스턴스가 계속 켜져 있으므로 과금이 발생합니다. 포트폴리오 API 테스트가 완전히 끝난 후에는 GCP 콘솔 접속 후 **[Vertex AI > 엔드포인트]** 로 이동해 해당 엔드포인트를 반드시 **삭제(Undeploy/Delete)** 해 주시기 바랍니다.