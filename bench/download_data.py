#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_data.py — 벤치마크 데이터 한 번에 받기 (download_data.sh 의 윈도우/파워셸 판; wget·unzip·bash 없이 파이썬만으로)

    cd mesh\\bench
    python download_data.py            # 전부 (e1 e2 e3 e4)
    python download_data.py e1 e3      # 골라서

받는 곳: mesh/bench/data/ (download_data.sh 와 같은 경로·파일명). 합계 약 60MB (BC5CDR·NCBI 제외). 인터넷만 있으면 5~10분.
필요한 것: 파이썬 3.9+, git (HealthVer · Check-COVID · 인과 언어 저장소), E2 과장 탐지는 pandas+pyarrow (없으면 안내만 하고 넘어감).
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = {"User-Agent": "komed-rag-bench/1.0 (download_data.py)"}


def ok(msg):
    print("  ✔", msg)


def no(msg):
    print("  ✘", msg)


def has(rel):
    p = os.path.join(DATA, rel)
    return os.path.isfile(p) and os.path.getsize(p) > 0


def fetch(url, dest=None, timeout=120):
    """URL → bytes (dest 가 있으면 파일로도 저장). 실패하면 None."""
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = io.BytesIO()
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk); total += len(chunk)
                print(f"\r    {url.rsplit('/', 1)[-1]} {total / 2**20:.1f}MB", end="", flush=True)
            print()
        data = buf.getvalue()
        if dest:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
        return data
    except Exception as e:
        print(f"\n    다운로드 실패 {url}\n    {type(e).__name__}: {str(e)[:160]}")
        return None


def git_clone(url, dest):
    if os.path.isdir(dest):
        return True
    try:
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, dest], check=True)
        return True
    except FileNotFoundError:
        no("git 이 없습니다 — https://git-scm.com 설치 후 다시"); return False
    except subprocess.CalledProcessError as e:
        no(f"git clone 실패 {url} (exit {e.returncode})"); return False


# ─────────────────────────────────────────────── E1  판정 계층
def e1():
    print("[E1] SciFact · HealthVer · Check-COVID")
    # SciFact — 1,409 claims · 5,183 초록 (CC BY-NC 2.0)
    if not has("scifact/claims_dev.jsonl"):
        data = fetch("https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz")
        if data:
            os.makedirs(os.path.join(DATA, "scifact"), exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for m in tf.getmembers():
                    parts = m.name.split("/")
                    if len(parts) < 2 or not m.isfile():
                        continue                              # 최상위 폴더 한 겹 벗김 (--strip-components=1)
                    rel = "/".join(parts[1:])
                    out = os.path.join(DATA, "scifact", rel)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with tf.extractfile(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    ok("scifact/claims_dev.jsonl") if has("scifact/claims_dev.jsonl") else no("SciFact — URL 이 바뀌었을 수 있다. github.com/allenai/scifact README 의 다운로드 링크 확인")

    # HealthVer — test 1,823 쌍
    if not has("healthver/data/healthver_test.csv"):
        git_clone("https://github.com/sarrouti/healthver", os.path.join(DATA, "healthver"))
    ok("healthver/data/healthver_test.csv") if has("healthver/data/healthver_test.csv") else no("HealthVer")

    # Check-COVID — SciFact 형식 (MIT)
    if not has("Check-COVID/README.md"):
        git_clone("https://github.com/posuer/Check-COVID", os.path.join(DATA, "Check-COVID"))
    if has("Check-COVID/README.md"):
        ok("Check-COVID (아래에서 주장·코퍼스 파일 경로 확인)")
        n = 0
        for root, _, files in os.walk(os.path.join(DATA, "Check-COVID")):
            for f in files:
                if f.endswith(".json") and any(k in f.lower() for k in ("test", "corpus")):
                    print("     ", os.path.relpath(os.path.join(root, f), DATA)); n += 1
                    if n >= 8:
                        break
    else:
        no("Check-COVID")


# ─────────────────────────────────────────────── E2  범위 초과
def e2():
    print("[E2] 과장 탐지 · 인과 언어")
    # 과장 탐지 — Wright & Augenstein 2021 (GPL-3.0). HF parquet → csv (pandas+pyarrow 필요)
    # HF 공개 분할: train 100 / test 563 (합 663). 논문 본문은 "553 로 시험" 이라 적어 10건 차이 — 문헌 MT-PET 값은 553 기준(각주로 밝힘).
    # 표 6 의 same 406 / exaggerates 144 / downplays 113 은 전체 663 의 분포이지 test 분포가 아니다 → test 분포는 로더가 센 값을 적는다.
    # train 100 은 문헌이 few-shot 학습에 쓴 분할 → 우리는 E2 예행(프롬프트·접기·규칙 점검) 전용. test 는 프롬프트 동결 뒤 한 번만.
    for split in ("test", "train"):
        csv = f"exagg_{split}.csv"
        if not has(csv):
            pq = os.path.join(DATA, f"exagg_{split}.parquet")
            if not os.path.isfile(pq):
                fetch(f"https://huggingface.co/api/datasets/copenlu/scientific-exaggeration-detection/parquet/default/{split}/0.parquet", pq)
            if os.path.isfile(pq):
                try:
                    import pandas as pd
                    pd.read_parquet(pq).to_csv(os.path.join(DATA, csv), index=False)
                except Exception as e:
                    no(f"parquet → csv 변환 실패 ({type(e).__name__}) — `pip install pandas pyarrow` 후 다시 실행하면 이 단계만 다시 돈다")
        ok(f"{csv} ({'본실행 563쌍' if split == 'test' else '예행 100쌍'})") if has(csv) \
            else no(f"과장 탐지 {split} — huggingface.co/datasets/copenlu/scientific-exaggeration-detection 에서 {split} 분할 수동 저장")

    # 인과 언어 — Yu et al. 2019 (GPL-3.0)
    repo = os.path.join(DATA, "causal-language-use-in-science")
    if not has("causal-language-use-in-science/README.md"):
        git_clone("https://github.com/junwang4/causal-language-use-in-science", repo)
    if not has("causal_pubmed.csv") and os.path.isdir(os.path.join(repo, "data")):
        csvs = sorted(f for f in os.listdir(os.path.join(repo, "data")) if f.endswith(".csv"))
        if csvs:
            shutil.copy2(os.path.join(repo, "data", csvs[0]), os.path.join(DATA, "causal_pubmed.csv"))
    ok("causal_pubmed.csv (라벨 분포 0:1,356 · 1:494 · 2:213 · 3:998 인지 아래에서 확인)") if has("causal_pubmed.csv") \
        else no("인과 언어 — data/ 안의 csv 를 causal_pubmed.csv 로 복사")


# ─────────────────────────────────────────────── E3  검색 계층
def e3():
    print("[E3] NFCorpus (BEIR)")
    # BEIR zip 을 써야 한다 — HF parquet 판은 corpus 의 PubMed url 이 지워져 있어 PMID→MeSH 를 못 붙인다
    if not has("nfcorpus/corpus.jsonl"):
        data = fetch("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip")
        if data:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(DATA)
    if has("nfcorpus/corpus.jsonl"):
        n_docs = sum(1 for _ in open(os.path.join(DATA, "nfcorpus", "corpus.jsonl"), encoding="utf-8"))
        n_q = sum(1 for _ in open(os.path.join(DATA, "nfcorpus", "queries.jsonl"), encoding="utf-8"))
        ok(f"nfcorpus/ ({n_docs} 문서 · {n_q} 질의)")
        with open(os.path.join(DATA, "nfcorpus", "corpus.jsonl"), encoding="utf-8") as f:
            head = f.read(200000)
        ok("corpus 에 url 필드 있음 → fetch_mesh_pmids.py 가능") if '"url"' in head else no("url 필드 없음 — BEIR zip 판이 맞는지 확인")
    else:
        no("NFCorpus")


# ─────────────────────────────────────────────── E4  개체 연결
def e4():
    print("[E4] BC5CDR · NCBI-disease")
    if not has("ncbi/NCBItestset_corpus.txt"):
        data = fetch("https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip")
        if data:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(os.path.join(DATA, "ncbi"))
    ok("ncbi/NCBItestset_corpus.txt") if has("ncbi/NCBItestset_corpus.txt") else no("NCBI-disease — ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/ 에서 수동")
    # BC5CDR — 계정 불필요: NCBI FTP 공개 (9/20 이미 넣어 둠)
    if not has("bc5cdr/CDR_TestSet.PubTator.txt"):
        data = fetch("https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip")
        if data:
            os.makedirs(os.path.join(DATA, "bc5cdr"), exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for m in zf.namelist():
                    if m.endswith(".PubTator.txt") or m.endswith("README.txt"):
                        with zf.open(m) as src, open(os.path.join(DATA, "bc5cdr", os.path.basename(m)), "wb") as dst:
                            shutil.copyfileobj(src, dst)
    ok("bc5cdr/CDR_TestSet.PubTator.txt") if has("bc5cdr/CDR_TestSet.PubTator.txt") \
        else no("BC5CDR — https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip → *.PubTator.txt 를 data/bc5cdr/ 에")


def check_loaders():
    print("\n[점검] 로더가 읽는지 · 라벨 분포가 문헌과 맞는지")
    py = sys.executable
    runs = [("scifact", ["data/scifact/claims_dev.jsonl", "data/scifact/corpus.jsonl"], "scifact/claims_dev.jsonl"),
            ("healthver", ["data/healthver/data/healthver_test.csv"], "healthver/data/healthver_test.csv"),
            ("exagg", ["data/exagg_test.csv"], "exagg_test.csv"),
            ("exagg", ["data/exagg_train.csv"], "exagg_train.csv"),
            ("causal", ["data/causal_pubmed.csv"], "causal_pubmed.csv")]
    for name, args, need in runs:
        if has(need):
            subprocess.run([py, "data_loaders.py", name] + args, cwd=HERE)
    print("\n기대값 — SciFact dev 쌍 450 안팎 · HealthVer 1,823 · 과장 test 563 + train 100 (test 의 same/exagg/down 분포는 여기 찍힌 값을 원고에 적는다;"
          " 문헌 표 6 의 406/144/113 은 전체 663 기준) · 인과 1,356/494/213/998")
    print("다르면: data_loaders.py 의 CAUSAL_CODE(인과 라벨 뜻) 또는 파일 경로를 확인.")


def main():
    what = [a.lower() for a in sys.argv[1:]] or ["e1", "e2", "e3", "e4"]
    os.makedirs(DATA, exist_ok=True)
    for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
        if name in what or (name == "e4" and "ncbi" in what):
            fn()
    check_loaders()


if __name__ == "__main__":
    main()
