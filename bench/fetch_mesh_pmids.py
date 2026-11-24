#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_mesh_pmids.py — BEIR corpus.jsonl(NFCorpus 등)의 PubMed URL 에서 PMID 를 뽑아 E-utilities 로 MeSH 표목(UI)을 받아 둔다.

    python fetch_mesh_pmids.py --corpus data/nfcorpus/corpus.jsonl --out data/nfcorpus/mesh.json [--api-key KEY --email you@x.org]

출력 mesh.json: {"MED-10": {"pmid": "12345678", "mesh": [{"ui": "D001943", "name": "Breast Neoplasms", "major": true}, ...], "pubtypes": [...], "year": "2004"}, ...}
NFCorpus 3,633 문서 → 200건씩 19회 efetch (키 없이 3회/초 → 1분 안). PMID 가 URL 에 없는 문서는 "pmid": null 로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMID_RE = re.compile(r"pubmed(?:\.ncbi\.nlm\.nih\.gov)?/(\d+)|[?&]term=(\d+)|/(\d{6,9})/?$")


def pmid_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = PMID_RE.search(url)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def efetch(pmids: list[str], api_key: str | None, email: str | None, tool: str = "komedrag-bench") -> dict:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": tool}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE, data=data), timeout=120) as r:
                xml = r.read()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    out = {}
    root = ET.fromstring(xml)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        mesh = []
        for mh in art.iter("MeshHeading"):
            d = mh.find("DescriptorName")
            if d is not None:
                mesh.append({"ui": d.get("UI"), "name": d.text, "major": d.get("MajorTopicYN") == "Y"})
        pts = [p.text for p in art.iter("PublicationType") if p.text]
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        out[pmid] = {"mesh": mesh, "pubtypes": pts, "year": year}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()
    docs = {}
    for line in open(a.corpus, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        url = (d.get("metadata") or {}).get("url") or d.get("url")
        docs[d["_id"]] = {"pmid": pmid_from_url(url), "mesh": [], "pubtypes": [], "year": ""}
    have = {k: v for k, v in docs.items() if v["pmid"]}
    print(f"문서 {len(docs)} · PMID 있음 {len(have)} · 없음 {len(docs) - len(have)}")
    done = {}
    if os.path.isfile(a.out):
        done = json.load(open(a.out, encoding="utf-8"))
        print(f"  기존 {sum(1 for v in done.values() if v.get('mesh'))}건 재사용")
    pending = [k for k, v in have.items() if not (done.get(k) or {}).get("mesh")]
    pmid2ids = {}
    for k in pending:
        pmid2ids.setdefault(have[k]["pmid"], []).append(k)
    pmids = list(pmid2ids)
    delay = 0.11 if a.api_key else 0.34
    for i in range(0, len(pmids), a.batch):
        chunk = pmids[i:i + a.batch]
        res = efetch(chunk, a.api_key, a.email)
        for p in chunk:
            for k in pmid2ids[p]:
                docs[k].update(res.get(p, {}))
        print(f"  {min(i + a.batch, len(pmids))}/{len(pmids)}  (받은 {len(res)})", flush=True)
        time.sleep(delay)
    for k, v in done.items():
        if v.get("mesh") and k in docs and not docs[k]["mesh"]:
            docs[k] = v
    json.dump(docs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    n_mesh = sum(1 for v in docs.values() if v["mesh"])
    print(f"→ {a.out}: MeSH 표목 있음 {n_mesh}/{len(docs)} · 문서당 평균 {sum(len(v['mesh']) for v in docs.values())/max(1,n_mesh):.1f}개")


if __name__ == "__main__":
    main()
