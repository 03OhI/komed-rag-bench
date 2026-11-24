"""Opt-in grounded reader and coherent-scope candidate. No benchmark labels are inputs."""
import copy

import verdict
from grounded_text import checked_call, integer, prepare_hits, quote_present

VERSION = 'grounded-coherent-v3-20261023'
GUIDANCE = """
추가 판독 절차:
1. 주장과 각 문헌의 개입/노출 및 결과지표를 맞춰라. 배경 문장과 연구가 실제 관찰한 결과를 구분한다.
2. 부정은 주장 방향에 상대적이다. 'X가 Y를 줄이지 않는다'는 주장에 '감소 없음'은 supports다.
   단순 비유의성이나 연구 부족이 효과 부재를 입증한다고 확대하지 말고 초록의 실제 결론을 따르라.
3. 효과 방향과 표현 강도를 분리한다. 상관관계 근거라도 같은 방향이면 supports로 읽고 인과 과장은 차원 코딩으로 다룬다.
4. may be associated는 상관, can be measured는 측정 가능성이다. 관계에 걸린 완화·부정만 인과 강도에 반영한다.
5. 서로 다른 연구의 최고 차원들을 합쳐 하나의 연구인 것처럼 판단하지 않는다.
6. 각 문헌을 정확히 한 번 판독하라. 관련 문헌은 주어진 초록 구간의 원문 인용과 모든 차원 코딩을 반드시 제출한다.
   […]는 중간 생략 표시이며 그 양쪽을 연결해 새 인용 문장을 만들면 안 된다.
7. 한국어 설명에는 사실 판독만 적고 최종 범주를 미리 단정하지 않는다. 출력 안의 문헌 번호는 실제 제공된 번호여야 한다.
"""

TOOL = copy.deepcopy(verdict.TOOL)
TOOL['name'] = 'submit_grounded_reading_v3'
SCHEMA = TOOL['input_schema']['properties']['evidence']['items']
SCHEMA['properties']['coding']['required'] = list(verdict.DIMS)
SCHEMA['required'] = ['n', 'relevant', 'stance', 'key_sentence', 'coding']


def validate(reading, hits):
    if not isinstance(reading, dict) or not isinstance(reading.get('evidence'), list):
        raise ValueError('Expected reading object and evidence array')
    for dim, meta in verdict.DIMS.items():
        integer(reading['claim_coding'][dim], 1, len(meta['levels']), 'claim.' + dim)
    seen = set()
    for e in reading['evidence']:
        n = integer(e['n'], 1, len(hits), 'evidence.n')
        if n in seen:
            raise ValueError(f'Duplicate evidence number {n}')
        seen.add(n)
        if type(e['relevant']) is not bool:
            raise ValueError('relevant must be boolean, not a string')
        if e['relevant']:
            if e.get('stance') not in ('supports', 'refutes', 'neutral'):
                raise ValueError(f'Invalid stance at {n}')
            if not quote_present(e.get('key_sentence'), hits[n-1]['_segments']):
                raise ValueError(f'Quote at {n} is absent from the shown abstract segments')
            for dim, meta in verdict.DIMS.items():
                integer(e['coding'][dim], 1, len(meta['levels']), f'evidence[{n}].' + dim)
    if seen != set(range(1, len(hits)+1)):
        raise ValueError('Every shown document must be read exactly once')
    if not isinstance(reading.get('explanation_ko'), str):
        raise ValueError('explanation_ko must be text')


def decide_coherent(claim, hits, reading, rules):
    """Keep stance weights/thresholds; require a single supporting scope witness.

    A human observational study plus an animal causal study does not establish
    a human causal effect merely by taking each dimension's maximum.
    """
    validate(reading, hits)
    result = verdict.decide(claim, hits, reading, rules)
    support = [e for e in result['evidence'] if e['relevant'] and e['stance'] == 'supports']
    profiles = []
    for e in support:
        gaps = [d for d in verdict.DIMS
                if result['claim_coding'][d] - e['coding'][d] >= rules['exceed_threshold'][d]]
        profiles.append({'n': e['n'], 'pmid': e['pmid'], 'exceeded_dims': gaps})
    result['scope_profiles'] = profiles
    if result['category'] == '지지' and profiles and all(p['exceeded_dims'] for p in profiles):
        # Minimal missing-scope witness; deterministic tie break in retrieval order.
        profile = min(profiles, key=lambda p: (len(p['exceeded_dims']), p['n']))
        witness = next(e for e in support if e['n'] == profile['n'])
        result['category'] = '범위 초과'
        result['sub'] = '서로 다른 연구의 차원별 최고값을 합쳐야만 지지 가능 — 단일 연구 범위를 초과'
        for d in profile['exceeded_dims']:
            meta = verdict.DIMS[d]; c = result['claim_coding'][d]; e = witness['coding'][d]
            result['exceeded'].append({'dim': d, 'name': meta['name'], 'claim_level': c,
                                      'claim_label': meta['levels'][c-1], 'evidence_level': e,
                                      'evidence_label': meta['levels'][e-1], 'pmid': witness['pmid'],
                                      'sentence': witness['key_sentence'], 'sentence_ko': witness['key_sentence_ko']})
    result['reader_version'] = VERSION
    return result


def judge(claim, hits, llm, rules=None):
    rules = rules or verdict.load_rules()
    shown = prepare_hits(hits, rules)
    if not shown:
        result = verdict.judge(claim, [], llm, rules)
        result.update(reader_version=VERSION, reading_audit={'calls': 0, 'repairs': []}, shown_evidence=[])
        return result
    prompt = verdict._prompt(claim, shown, rules).replace('submit_reading', TOOL['name'])
    try:
        reading, audit = checked_call(llm, verdict.SYSTEM + '\n' + GUIDANCE, prompt, TOOL,
                                     lambda out: validate(out, shown))
        result = decide_coherent(claim, shown, reading, rules)
        result.update(llm=llm.label, reading_audit=audit,
                      shown_evidence=[{'pmid': h.get('pmid'), 'segments': h['_segments']} for h in shown])
        return result
    except Exception as exc:
        result = verdict.judge(claim, [], llm, rules)
        result.update(nei_reason='error', sub='판독 실패 — 재시도 필요', error=str(exc)[:500],
                      llm=llm.label, reader_version=VERSION)
        return result
