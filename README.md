# KoMed-RAG 실험 코드

실험 결과보고서의 실험 1~4를 돌리고 채점한 코드입니다. 폴더 구조는 실행하던 그대로 두었습니다(`bench/`의 스크립트가 상위 폴더의 모듈을 불러옵니다).

## 실험별 파일

### 실험 1. 규칙 유무 비교 (HealthVer 1,823쌍)

| 파일 | 하는 일 |
|---|---|
| `bench/run_verdict.py` | 실행기. `--cond ours`(KoMed-RAG) · `free`(LLM 직접 판정) · `norule`(규칙을 뺀 KoMed-RAG) |
| `verdict.py`, `verdict_rules.json` | 판독 프롬프트와 고정 판정 규칙 |
| `bench/data_loaders.py`, `bench/download_data.py` | 데이터 받기와 로더 |
| `bench/llm_client.py`, `bench/batch_gemini.py`, `komed_llm.py` | LLM 호출·캐시, Gemini Batch 제출과 수거 |
| `bench/metrics.py`, `bench/report.py` | 지표와 집계 |
| `bench/report_e1_final.py` | 최종 표(진실성 점수, 군집 부트스트랩 2,000회) |
| `bench/report_e1_model.py`, `bench/run_model.py` | 판독 LLM을 바꾼 표(GPT-4o 행) |
| `bench/e1_pooled_1113.py` | HealthVer + Check-COVID 합산 분석 |
| `bench/test_norule.py` | 규칙을 뺀 조건의 단위 시험 |

결과: `bench/preds_e1_1113/report_e1.md`, `report_e1_pooled.md`, `bench/preds_e1_gpt4o/report_e1_gpt4o.md`

### 실험 2. 과장 탐지 (Wright & Augenstein 시험 563쌍)

| 파일 | 하는 일 |
|---|---|
| `bench/run_exagg.py` | 실행기. `ours4`(KoMed-RAG) · `direct`(LLM 판정) · `ours7` · `directreview` · `aligned4` |
| `bench/compare_exagg.py` | 방식별 점수와 짝지은 차이 |
| `bench/make_exagg_swap.py` | 두 문장의 자리를 바꾼 자료 만들기 |
| `bench/swap_exagg.py` | 위치 교환 검사 집계(뒤집기 일치율, 자리 기울기) |
| `bench/test_compare_exagg.py`, `bench/test_swap_exagg.py` | 단위 시험 |

결과: `bench/preds_exagg_test/report_exagg_test.md`, `bench/preds_exagg_swap/report_swap_test.md`

### 실험 3. 검색 비교 (NFCorpus)

| 파일 | 하는 일 |
|---|---|
| `bench/fetch_mesh_pmids.py` | NFCorpus 문서에 MeSH 표목 붙이기 |
| `bench/run_nfcorpus.py` | `bm25` · `bm25_mesh` · `gate` · `gate_rrf` |
| `bench/e3_gate_diag.py`, `bench/e3_summary.py` | 게이트 단독 진단과 요약 |
| `retrieve.py`, `build_pubmed_index.py`, `bm25_util.py` | 시스템 검색 모듈과 색인 |
| `bench/run_system_on_nfcorpus.py` | 시스템 검색 모듈을 NFCorpus에서 재현 |
| `medcpt_rerank.py`, `bench/rerank_nfcorpus.py` | MedCPT 재순위와 채점 |
| `bench/E3_rerank_colab.ipynb` | 코랩 실행용 노트북 |

결과: `bench/report_ir_nfcorpus_1102.md`, `report_ir_nfcorpus_system_1102.md`, `report_ir_nfcorpus_rerank_medcpt_top30.md`

### 실험 4. 한국어 용어 → MeSH (XL-BEL)

| 파일 | 하는 일 |
|---|---|
| `bench/run_xlbel_ko.py` | 실행과 채점 |
| `ko_resolver.py`, `ko_backoff.py`, `mesh_resolve.py` | 리졸버와 층위 후퇴 |
| `ko_alias.json`, `ko_stoplist.txt`, `build_ko_alias.py` | 한국어 별칭 사전과 만드는 스크립트 |
| `bench/analyze_xlbel_errors.py` | 무응답과 오답 분해 |

결과: `bench/report_el_xlbel_ko.md`, `bench/xlbel_error_analysis_kob.md`

그 밖의 `.py`는 위 스크립트가 불러오는 모듈입니다.

## 저장소에 없는 것

- API 키: 환경변수나 `<provider>_key.txt`로 따로 둡니다.
- 벤치마크 데이터(`bench/data/`): `bench/download_data.py`로 받습니다. `data/xlbel/cui2mesh.json`은 UMLS에서 만든 대응표여서 올리지 않았습니다.
- 항목별 예측 파일(`*.jsonl`)과 LLM 캐시, PubMed 색인, `mesh_ko_full.json` 같은 대용량 파일.

## 필요한 라이브러리

`numpy`, `pandas`, `rapidfuzz`, `torch`, `transformers`(재순위에만 필요)

## 라이선스

이 저장소의 자체 저작물에는 [GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0-only`)가 적용됩니다. Copyright (c) 2026 Mun Yeeun (문예은). 외부 벤치마크 데이터와 모델에는 각 배포처의 라이선스가 적용됩니다.
