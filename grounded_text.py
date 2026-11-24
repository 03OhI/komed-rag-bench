"""Text-only evidence handling shared by candidate readers; never accepts gold labels."""
import copy
import json
import re


def normalize(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def quote_present(quote, segments):
    quote = normalize(quote)
    return bool(quote) and any(quote in normalize(segment) for segment in segments)


def evidence_view(abstract, budget):
    """Keep beginning and conclusion under the original character budget.

    A visible omission marker prevents the two ranges from appearing contiguous.
    Quotes are checked separately against each range, never across the gap.
    """
    text = normalize(abstract)
    if budget < 40:
        raise ValueError('Evidence character budget must be at least 40')
    if len(text) <= budget:
        return text, [text]
    marker = ' […] '
    head_budget = (budget - len(marker)) // 2
    tail_budget = budget - len(marker) - head_budget
    head, tail = text[:head_budget], text[-tail_budget:]
    # Prefer complete sentences without inventing or rewriting source text.
    boundary = list(re.finditer(r'[.!?]\s+', head))
    if boundary and boundary[-1].end() >= head_budget // 2:
        head = head[:boundary[-1].start() + 1]
    boundary = re.search(r'[.!?]\s+', tail)
    if boundary and boundary.end() < tail_budget // 2:
        tail = tail[boundary.end():]
    segments = [head.strip(), tail.strip()]
    return marker.join(segments), segments


def prepare_hits(hits, rules):
    """Deduplicate PMID (or identical title+abstract), then allocate evidence slots."""
    unique, seen = [], set()
    for h in hits:
        key = ('pmid', str(h['pmid'])) if h.get('pmid') else ('text', normalize(h.get('title')), normalize(h.get('abstract')))
        if key in seen:
            continue
        seen.add(key)
        view, segments = evidence_view(h.get('abstract'), rules['abstract_chars'])
        unique.append({**h, 'abstract': view, '_segments': segments})
        if len(unique) == rules['k_evidence']:
            break
    return unique


def checked_call(llm, system, prompt, tool, validate, max_calls=2):
    """One bounded repair on structural/grounding failure; no gold or scores available.

    A transport failure is propagated, never hidden as a successful abstention.
    """
    audit = []
    current = prompt
    for attempt in range(max_calls):
        out = llm.call(system, current, tool)
        try:
            validate(out)
            return copy.deepcopy(out), {'calls': attempt + 1, 'repairs': audit}
        except (ValueError, TypeError, KeyError) as exc:
            audit.append(str(exc))
            if attempt + 1 == max_calls:
                raise ValueError('Grounded reading invalid after repair: ' + str(exc)) from exc
            current = (prompt + '\n\n이전 출력의 아래 오류만 원문을 확인하여 수정하라. 정답 라벨은 제공되지 않는다. '
                       '주어진 원문 전체를 다시 확인하고 완전한 객체를 제출하라.\n검증 오류: ' + str(exc) +
                       '\n이전 출력(JSON 데이터): ' + json.dumps(out, ensure_ascii=False))


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value
