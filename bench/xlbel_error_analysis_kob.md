# XL-BEL KO 오류 분해 — kob (610건, MeSH 대응 가능분)

## 1. 엄격 P@1 vs 계층 관대 P@1 (gold 또는 그 부모·자식)

| 트리 | n | 엄격 P@1 | +부모/자식 | +형제 | 무응답 | 진짜 오답 |
|---|---|---|---|---|---|---|
| C | 163 | 32.5 | 43.6 | 44.8 | 41.7 | 13.5 |
| D | 75 | 45.3 | 56.0 | 56.0 | 37.3 | 6.7 |
| A | 38 | 31.6 | 36.8 | 36.8 | 42.1 | 21.1 |
| B | 300 | 21.7 | 28.0 | 28.7 | 62.0 | 9.3 |
| ALL | 610 | 28.9 | 36.6 | 37.2 | 51.8 | 11.0 |

관계 정의: 부모/자식 = MeSH 트리 번호가 한 단계 위/아래 · 형제 = 같은 부모. 예측이 gold 보다 **구체적**(자식)인 경우가 많다 — 앵커 표면형은 구체어인데 gold 는 링크된 문서의 상위 개념.

## 2. 오답의 관계 분포 (응답했으나 엄격 오답)

| 트리 | 자식(더 구체적) | 부모(더 일반적) | 형제 | 2단계 이상 | 무관 |
|---|---|---|---|---|---|
| C | 8 | 10 | 2 | 4 | 18 |
| D | 3 | 5 | 0 | 0 | 5 |
| A | 1 | 1 | 0 | 0 | 8 |
| B | 9 | 10 | 2 | 4 | 24 |

## 3. 무응답 원인

| 트리 | 원인 | n | 예 |
|---|---|---|---|
| C | 사전에 없음(용어·이형태) | 53 | 다리 이중골절 · 심실잔떨림 · 역병 · 육손이 · 한센인 · 죽상동맥경화증 |
| C | 구(句)·부가어·조사 포함 | 12 | 검푸른 피부 빛 · 뎅기열 발생을 · 빠른 심장 박동 · 2003년 사스 범유행 · 후천성 소아 녹내장 · 초기 급성 감염 |
| C | 스톱리스트(일반어 차단) | 3 | 부상 · 종양 · 종양 |
| D | 사전에 없음(용어·이형태) | 18 | 쇠 · 마그네슘선 · 취화물 · 무쇠 · 순금을 주 · 칼슘 이온 |
| D | 라틴 문자·원소기호(한국어 리졸버 범위 밖) | 9 | Mo · Oxalic acid · ethylenediaminetetraacetic acid(EDTA) · A · Ni · Se |
| D | 한글+라틴 혼용 | 1 | 철(II) |
| A | 사전에 없음(용어·이형태) | 16 | 우심실 · 대뇌 반구 · 겉귀 · 내분비 · 추골 · 추절흔 |
| B | 사전에 없음(용어·이형태) | 171 | 장어목 · 지빠귀과 · 들쭉 · 사초 · 절족동물 · 산자고속 |
| B | 구(句)·부가어·조사 포함 | 10 | 거농원숭이 · 구세계 원숭이 · 검둥원숭이 · 아베로아과 · 가시두더지과 · 거세되지 않은 숫양 |
| B | 라틴 문자·원소기호(한국어 리졸버 범위 밖) | 5 | orang utan · S. pneumoniae" · Salmonella enteritidis · eukaryote · maize |

## 4. '무관' 오답 전부 (사람이 볼 것)

**C** (18건)
- 심장 박동수 증가 → Heart Rate [particle>ko_suffix 89.0] · gold Tachycardia
- MERS CoV → Middle East Respiratory Syndrome Coronavirus [ko_exact 100.0] · gold Coronavirus Infections
- 간질병 → Fasciola hepatica [ko_suffix 92.0] · gold Epilepsy
- 안구 건조 → Desiccation [ko_head 89.0] · gold Xerophthalmia
- 매독 증상 → Signs and Symptoms [ko_head 89.0] · gold Chancre
- 물뇌증 → Anencephaly [ko_fuzzy 93.3] · gold Hydrocephalus
- 어지러움 → Vertigo [ko_exact 100.0] · gold Dizziness
- 음수증 → Drinking [ko_suffix 92.0] · gold Renal Insufficiency
- 칸디다 → Candida [ko_exact 100.0] · gold Candidiasis, Vulvovaginal
- 안구질환 → Eye [ko_suffix 92.0] · gold Eye Diseases
- 근육 → Muscles [ko_exact 100.0] · gold Myalgia
- 복부 대동맥 → Aorta, Abdominal [ko_exact 100.0] · gold Aortic Aneurysm, Abdominal
- 비늘증 → Animal Scales [ko_suffix 92.0] · gold Ichthyosis, Ichthyosis, Lamellar
- 배앓이 → Colic [ko_exact 100.0] · gold Abdominal Pain
- 해수병 → Seawater [ko_suffix 92.0] · gold Asthma
- 트리파노소마 → Trypanosoma [ko_exact 100.0] · gold Trypanosomiasis
- 근육 경련 → Muscle Cramp [ko_exact 100.0] · gold Seizures
- 언어 → Language [ko_exact 100.0] · gold Language Disorders

**D** (5건)
- 중이온 → Heavy Ions [ko_exact 100.0] · gold Ions
- 수분 → Pollination [ko_exact 100.0] · gold Water
- 소금물 → Saline Waters [ko_exact 100.0] · gold Water
- 산화마그네슘 → Magnesium Oxide [ko_exact 100.0] · gold Magnesium
- 수자원 → Water Resources [ko_exact 100.0] · gold Water

**A** (8건)
- 허파호흡 → Respiration [ko_head 89.0] · gold Respiratory System
- 피질 → Sebum [ko_fuzzy 88.9] · gold Adrenal Cortex
- 이막 → Forehead [ko_fuzzy 88.9] · gold Cochlea
- 호흡 → Respiration [ko_exact 100.0] · gold Respiratory System
- 대뇌각 → Cerebral Peduncle [ko_exact 100.0] · gold Cerebrum
- 내피세포 → Endothelial Cells [ko_exact 100.0] · gold Endothelium
- 혈장단백질 → Proteins [ko_head 89.0] · gold Plasma
- 임파 → Forehead [ko_fuzzy 88.9] · gold Lymph

**B** (24건)
- 인류 → Phthiraptera [ko_fuzzy 88.9] · gold Humans
- 사마귀 → Warts [ko_exact 100.0] · gold Mantodea
- 이끼류 → Lichens [ko_exact 100.0] · gold Bryophyta
- 폐렴구균 감염증 → Pneumococcal Infections [ko_exact 100.0] · gold Streptococcus pneumoniae
- 타마린속 → Tamarindus [ko_fuzzy 90.9] · gold Saguinus
- 차나무목 → Fagales [ko_fuzzy 94.7] · gold Theaceae
- 인 → Phosphorus [ko_exact 85.0] · gold Humans
- 블라스토키스티스강 → Blastocystis Infections [ko_fuzzy 88.4] · gold Blastocystis
- 저자 → Electrons [ko_fuzzy 88.9] · gold Humans
- 부유 → Infant Formula [ko_fuzzy 88.9] · gold Ephemeroptera
- 성성이 → Sex [particle>ko_variant 74.0] · gold Pongo
- 웨일 → Whey [ko_fuzzy 88.9] · gold Cetacea
- 관다발 → Plant Vascular Bundle [ko_exact 100.0] · gold Tracheophyta
- 인체 → Human Body [ko_exact 100.0] · gold Humans
- 단풍나무과 → Aceraceae [ko_exact 100.0] · gold Acer
- 차 → Tea [ko_exact 100.0] · gold Camellia sinensis
- 장내 미생물 → Gastrointestinal Microbiome [ko_fuzzy 89.7] · gold Enterobacteriaceae
- rose → Rapid On-site Evaluation [ko_exact 100.0] · gold Rosa
- 쥐여우원숭이 → Lemur [ko_fuzzy 92.3] · gold Cheirogaleidae
- 살모넬라 감염증 → Salmonella Infections [ko_exact 100.0] · gold Salmonella
- 자문 → Writing [ko_fuzzy 90.9] · gold Syzygium
- 젖산 → Lactic Acid [ko_exact 100.0] · gold Lactobacillus
- 독자 → Only Child [ko_exact 100.0] · gold Humans
- 바리 → Dissection [ko_fuzzy 88.9] · gold Bass

