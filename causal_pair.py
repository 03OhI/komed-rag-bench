"""E2 candidate: align propositions and ground each strength code in its source clause."""
from grounded_text import checked_call, integer, quote_present

GUIDANCE = """
두 문장에서 같은 노출/개입과 결과의 관계를 먼저 식별하라. 더 강한 단어가 있다는 이유만으로
배경 연구·방법·다른 결과지표의 표현을 가져오지 말고, 현재 결론의 핵심 관계를 코딩하라.
부정·완화 표현은 해당 관계에 걸리는 것만 적용한다. 'may be associated'는 상관이며
'can be measured'는 측정 가능성이지 개입의 인과 가능성이 아니다. 권고 문장을 인과 효과로 바꾸지 않는다.
관계가 없다고 보고한 결과와, 관계가 있는지 아직 모른다는 한계 진술을 구분한다.
관찰 연구라도 문장이 단정적 인과를 말하면 표현 강도대로 코딩한다. 연구설계를 보고 낮춰 쓰지 않는다.
excerpt에는 판단에 사용한 절을 해당 문장에서 그대로 복사한다. 관계 진술이 없으면 원문 문장 자체를 복사한다.
reason은 선택한 절과 강도 사이의 짧은 설명만 쓴다. 두 문장의 노출·결과가 실제로 다르면 aligned=false로 기록하되
벤치마크 정의에 따라 각 문장의 핵심 관계 강도를 독립적으로 코딩한다. 이 항목을 채점에서 제외하지 않는다.
"""

SIDE = {'type': 'object', 'properties': {
    'strength': {'type': 'integer', 'minimum': 0, 'maximum': 3},
    'exposure': {'type': 'string'}, 'outcome': {'type': 'string'},
    'excerpt': {'type': 'string'}, 'reason': {'type': 'string'}},
    'required': ['strength', 'exposure', 'outcome', 'excerpt', 'reason']}
TOOL = {'name': 'code_aligned_pair', 'description': '원문 절에 근거한 두 문장의 인과 강도',
        'input_schema': {'type': 'object', 'properties': {
            'press': SIDE, 'abstract': SIDE, 'aligned': {'type': 'boolean'}},
            'required': ['press', 'abstract', 'aligned']}}


def predict(press, abstract, llm, scale_system):
    prompt = f'[보도자료 결론]\n{press}\n[논문 초록 결론]\n{abstract}\n두 원문의 표현 강도를 제출하라.'

    def validate(out):
        if not isinstance(out, dict):
            raise ValueError('Expected aligned pair object')
        if type(out.get('aligned')) is not bool:
            raise ValueError('aligned must be boolean')
        for side, source in [('press', press), ('abstract', abstract)]:
            part = out[side]
            integer(part['strength'], 0, 3, side + '.strength')
            if not quote_present(part['excerpt'], [source]):
                raise ValueError(side + '.excerpt is not present in its own source')
            for field in ('exposure', 'outcome', 'reason'):
                if not isinstance(part[field], str):
                    raise ValueError(side + '.' + field + ' must be text')

    out, audit = checked_call(llm, scale_system + '\n' + GUIDANCE, prompt, TOOL, validate)
    s1, s2 = out['press']['strength'], out['abstract']['strength']
    return {'s1_4': s1, 's2_4': s2,
            'pred': 'exaggerates' if s1 > s2 else ('downplays' if s1 < s2 else 'same'),
            'aligned_reading': out, 'reading_audit': audit}
