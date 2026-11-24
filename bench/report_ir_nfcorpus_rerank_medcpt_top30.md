# E3 재순위 — nfcorpus (질의 323, 그중 시스템 순위가 빈 질의 14 · 재순위 상위 30 · ncbi/MedCPT-Cross-Encoder · max_len 512)

하네스 BM25: nDCG@10 0.326 · Recall@100 0.246

| 조건 | nDCG@10 [95% CI] | nDCG@3 | Recall@100 | Δ nDCG@10 vs BM25 [95% CI] | Δ nDCG@10 vs 시스템 게이트+RRF [95% CI] | Δ nDCG@3 vs 시스템 [95% CI] |
|---|---|---|---|---|---|---|
| 시스템 게이트+RRF (재순위 없음) | 0.320 [0.284, 0.356] | 0.371 | 0.277 | -0.006 [-0.026, +0.014] | — | — |
| + 재순위 · 질의 raw | 0.347 [0.310, 0.383] | 0.411 | 0.277 | +0.021 [+0.003, +0.039] | +0.027 [+0.010, +0.045] | +0.041 [+0.013, +0.068] |
| ↳ 질의 출처 | {'mesh': 0, 'raw': 323} | | | | | |

재순위는 상위 30건 안의 순서만 바꾸므로 Recall@30 은 그대로다. 질의 raw = 영어 원문(한국어 기사라면 LLM 번역이 있을 때), mesh = MeSH 표목명 키워드(한국어 기사에서 LLM 없이 갈 때, 개념 0건이면 원문). 재순위는 MeSH 효과가 아니라 시스템 개선으로 보고한다.
점수 캐시: rerank_cache_medcpt.jsonl (48,327쌍) · 236초
→ /content/drive/MyDrive/KoMedRAG_E3_rerank/bench/report_ir_nfcorpus_rerank_medcpt_top30.md

<!-- 2026-11-09 14:41 KST 코랩(졸작 계정 드라이브)에서 --split test --variants raw --top 30 --scorer medcpt --final 1회 실행. 화면 출력을 그대로 옮김. trec 파일은 드라이브에만 있음 -->
