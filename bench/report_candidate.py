"""Paired candidate comparisons, including failures in the denominator."""
from collections import defaultdict
from statistics import mean
from metrics import macro_f1, accuracy, paired_bootstrap_diff


def table_candidate(dataset, groups, lines):
    four = dataset == 'e6'
    gk, pk = ('gold4', 'pred4') if four else ('gold', 'pred')
    labels = ('SUPPORT', 'EXCEEDS', 'REFUTE', 'NEI') if four else ('same', 'exaggerates', 'downplays')
    comparisons = ([('ours', 'oursv3'), ('stdrag', 'oursv3'), ('stdragv3', 'oursv3'), ('gatefreev3', 'oursv3')]
                   if four else [('ours7', 'aligned4'), ('ours4', 'aligned4'), ('direct', 'aligned4'), ('directreview', 'aligned4')])
    names = {name for pair in comparisons for name in pair}
    if not any(c in ('oursv3', 'aligned4') for c, _ in groups):
        return
    lines += [f'\n### {dataset} 개선판의 동일 항목 비교\n',
              '두 조건에서 시도했고 정답이 등록된 공통 항목으로 계산한다. API/형식 실패는 ERROR 예측으로 남겨 오답에 포함한다. '
              'CI는 항목 단위 짝지은 부트스트랩 95%; 주장 중복·군집이 있으면 확증 추론에 그대로 사용할 수 없다. '
              '차이의 구간이 0을 포함하면 우월성을 확인한 것으로 해석하지 않는다.',
              '| 모델 | 기준선 → 개선판 | 공통/합집합 | Δ macro-F1 [CI] | Δ 정확도 [CI] | 오류 기준선/개선판 |',
              '|---|---|---|---|---|---|']
    by_llm = defaultdict(dict)
    for (cond, llm), rows in groups.items():
        by_llm[llm][cond] = rows
    for llm, conds in sorted(by_llm.items()):
        for base, candidate in comparisons:
            if base not in conds or candidate not in conds:
                continue
            aa, bb = ({r['id']: r for r in conds[c]} for c in (base, candidate))
            if len(aa) != len(conds[base]) or len(bb) != len(conds[candidate]):
                raise ValueError('Duplicate IDs in candidate comparison')
            common = sorted(aa.keys() & bb.keys())
            if any(aa[i].get(gk) != bb[i].get(gk) for i in common):
                raise ValueError('Candidate comparison gold mismatch')
            common = [i for i in common if aa[i].get(gk) in labels]
            if not common:
                continue
            pred = lambda r: r[pk] if not r.get('error') and r.get(pk) in labels else 'ERROR'
            g, p, q = [aa[i][gk] for i in common], [pred(aa[i]) for i in common], [pred(bb[i]) for i in common]
            metric = lambda g, p: macro_f1(g, p, labels)
            df, da = metric(g,q)-metric(g,p), accuracy(g,q)-accuracy(g,p)
            cf, ca = paired_bootstrap_diff(g,p,q,metric), paired_bootstrap_diff(g,p,q,accuracy)
            lines.append(f'| {llm} | {base} → {candidate} | {len(common)}/{len(aa.keys() | bb.keys())} | '
                         f'{df:+.3f} [{cf[0]:+.3f}, {cf[1]:+.3f}] | {da:+.3f} [{ca[0]:+.3f}, {ca[1]:+.3f}] | '
                         f'{p.count("ERROR")}/{q.count("ERROR")} |')
    lines += ['\n| 모델 | 조건 | 항목 수 | 평균 논리 호출 수 | 평균 판정 초 |', '|---|---|---|---|---|']
    for (cond,llm), rows in sorted(groups.items()):
        if cond not in names:
            continue
        calls = [r['llm_calls'] for r in rows if 'llm_calls' in r]
        secs = [r['sec'] for r in rows if 'sec' in r]
        cost = f'{mean(calls):.2f}' if calls else '—'
        seconds = f'{mean(secs):.3f}' if secs else '—'
        lines.append(f'| {llm} | {cond} | {len(rows)} | {cost} | {seconds} |')
    lines.append('논리 호출 수에는 캐시 적중도 포함하며 공급자 내부 재시도 수와 다르다. 실제 토큰 사용량은 bench_usage.jsonl을 확인한다. '
                 'E6 시간은 검색을 제외한 판정 시간이다. 비용이 다른 조건의 점수 차이를 같은 비용의 우세로 표현하지 않는다.')
