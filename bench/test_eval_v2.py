"""Regression tests for scoring validity, fair evidence and reproducible run boundaries."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager
from uuid import uuid4
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import protocol
import report
import report_safety
import report_entity_tradeoff
import run_verdict
import run_exagg
import run_e6_v2 as e6
import run_nfcorpus as ir
import style_compare_report
import verdict


@contextmanager
def workspace_temp():
    # Windows sandbox cannot use tempfile's owner-only (0700) directory ACL.
    # Keep small fixtures under bench for inspection; no recursive deletion.
    root = Path(__file__).resolve().parent / '_test_outputs'
    path = root / uuid4().hex
    path.mkdir(parents=True, mode=0o777)
    yield str(path)


class FakeLLM:
    label = "mock:test"

    def __init__(self, label="EXCEEDS", fail=False):
        self.answer, self.fail, self.prompts = label, fail, []

    def call(self, system, prompt, tool):
        self.prompts.append((system, prompt, tool))
        if self.fail:
            raise RuntimeError("simulated transport failure")
        if tool['name'] == 'submit_reading':
            return {"claim_coding": {"causal": 7, "advice": 1, "species": 3, "specificity": 1},
                    "evidence": [{"n": 1, "relevant": True, "stance": "supports",
                                  "key_sentence": "Coffee was associated with lower risk.",
                                  "coding": {"causal": 3, "advice": 1, "species": 3, "specificity": 1}}],
                    "explanation_ko": "원문은 상관관계를 보고합니다."}
        return {"label": self.answer, "reason": "fixture", "key_sentence": "Coffee was associated with lower risk.",
                "citations": [{"n": 1, "key_sentence": "Coffee was associated with lower risk."}]}


def record(i, g, p, **kwargs):
    return {"id": str(i), "gold": g, "pred": p, **kwargs}


class MetricsTests(unittest.TestCase):
    def test_distribution_and_failures(self):
        s = metrics.outcome_summary(["SUPPORT", "REFUTE", "NEI", "SUPPORT", "NEI"],
                                    ["SUPPORT", "NEI", "SUPPORT", "ERROR", "NEI"])
        self.assertEqual((s['correct'], s['missing'], s['wrong'], s['errors']), (2, 1, 1, 1))
        self.assertEqual(s['utility_attempted'], .2)
        self.assertEqual(s['selective_accuracy'], .5)

    def test_abstain_all_is_not_perfect_selective_prediction(self):
        s = metrics.outcome_summary(['SUPPORT','REFUTE','NEI'], ['NEI']*3)
        self.assertEqual(s['coverage'], 0)
        self.assertIsNone(s['selective_accuracy'])
        self.assertEqual(s['answerable_correct'], 0)

    def test_empty_is_undefined(self):
        self.assertIsNone(metrics.outcome_summary([], [])['correct_rate'])

    def test_error_on_gold_nei_gets_no_credit(self):
        s = metrics.outcome_summary(['NEI'], ['ERROR'])
        self.assertEqual(s['correct'], 0)
        self.assertEqual(s['errors'], 1)

    def test_four_labels_preserve_exaggeration_error(self):
        s = metrics.outcome_summary(['EXCEEDS'], ['SUPPORT'], metrics.LABELS4)
        self.assertEqual(s['wrong'], 1)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            metrics.outcome_summary(['NEI'], [])

    def test_paired_transitions_reveal_lost_correct_answer(self):
        a = [record(1,'SUPPORT','REFUTE'), record(2,'SUPPORT','SUPPORT')]
        b = [record(1,'SUPPORT','NEI'), record(2,'SUPPORT','NEI')]
        s = metrics.paired_outcome_changes(a,b)
        self.assertEqual((s['wrong_to_abstain'], s['correct_to_abstain']), (1,1))
        self.assertEqual(s['retained_correct_non_nei'], 0)

    def test_gold_mismatch_not_silently_paired(self):
        with self.assertRaises(ValueError):
            metrics.aligned_success([record(1,'NEI','NEI')], [record(1,'SUPPORT','NEI')])

    def test_pair_error_and_missing_ids_excluded_transparently(self):
        a = [record(1,'NEI','NEI'), record(2,'NEI','NEI')]
        b = [record(1,'NEI','NEI',error='failed')]
        s = metrics.paired_outcome_changes(a,b)
        self.assertEqual((s['n'], s['union_n']), (0,2))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rules = verdict.load_rules()
        self.hits = [{'pmid':'1','title':'Hidden title', 'abstract':'Coffee was associated with lower risk.', 'pubtypes':[]}]

    def test_quote_presence_is_not_guaranteed(self):
        self.assertEqual(run_verdict.citation_checks(['Invented quote'], self.hits, self.rules), [False])

    def test_title_not_an_abstract_quote(self):
        self.assertEqual(run_verdict.citation_checks(['Hidden title'], self.hits, self.rules), [False])

    def test_unshown_tail_is_not_credit(self):
        rules = {**self.rules, 'abstract_chars': 6}
        self.assertEqual(run_verdict.citation_checks(['lower risk.'], self.hits, rules), [False])

    def test_whitespace_normalization_and_document_number(self):
        q = ['Coffee  was associated with lower risk.']
        self.assertEqual(run_verdict.citation_checks(q, self.hits, self.rules, [1]), [True])
        self.assertEqual(run_verdict.citation_checks(q, self.hits, self.rules, [2]), [False])

    def test_invalid_free_label_and_transport_are_errors(self):
        item = {'id':'a','dataset':'healthver','claim':'test','gold':'NEI','docs':[]}
        for llm in [FakeLLM('BAD'), FakeLLM(fail=True)]:
            r = run_verdict.run_item(item, 'free', llm, self.rules)
            self.assertEqual(r['pred'], 'ERROR')
            self.assertIn('error',r)

    def test_e6_factorial_same_input_scope_label(self):
        c = {'id':'a','claim':'Coffee prevents cancer.','gold':'SUPPORT','gold4':'EXCEEDS','expected_raw':'범위 초과'}
        std = {'a': self.hits}
        a = e6.run_one(c,'stdrag',FakeLLM(),self.rules,std,std)
        b = e6.run_one(c,'bm25rules',FakeLLM(),self.rules,std,std)
        self.assertEqual(a['input_pmids'],b['input_pmids'])
        self.assertEqual((a['pred4'],b['pred4']),('EXCEEDS','EXCEEDS'))
        self.assertEqual((a['pred'],b['pred']),('SUPPORT','SUPPORT'))
        self.assertTrue(a['verbatim'][0])
        self.assertTrue(b['verbatim'][0])

    def test_nested_production_error_not_nei_success(self):
        sysm = SimpleNamespace(verify_claim=lambda _: {'fact':{'category':'근거 없음','nei_reason':'error','error':'transport'}})
        c = {'id':'a','claim':'x','gold':'NEI','gold4':'NEI','expected_raw':'근거 없음'}
        r = e6.run_one(c,'ours',FakeLLM(),self.rules,{}, {},sysm)
        self.assertEqual(r['pred4'],'ERROR')

    def test_e2_bad_label_not_majority_class(self):
        item = {'id':'a','gold':'same','pr':'x','abs':'y','pr_strength':1,'abs_strength':1}
        self.assertEqual(run_exagg.run_exagg_item(item,'direct',FakeLLM('BAD'))['pred'],'ERROR')


class ProtocolTests(unittest.TestCase):
    def test_e6_six_condition_cli_and_report(self):
        with workspace_temp() as d:
            root = Path(d); preds = root/'preds'; claims = root/'claims.tsv'
            claims.write_text('id\tclaim\texpected\nq\tCoffee prevents cancer.\t범위 초과\n',encoding='utf-8')
            hits = [{'pmid':'1','title':'coffee','abstract':'Coffee was associated with lower risk.','pubtypes':[]}]
            retr = SimpleNamespace(search=lambda claim,k: {'hits':hits})
            def make_system(*args):
                fact = verdict.decide('Coffee prevents cancer.',hits,FakeLLM().call('', '', {'name':'submit_reading'}),verdict.load_rules())
                return SimpleNamespace(verify_claim=lambda _: {'fact':fact,'evaluator':{'verdict':'correct'}})
            args = ['run_e6.py','--llm','mock','--claims',str(claims),'--out',str(preds)]
            with patch.object(sys,'argv',args), patch.dict(sys.modules,{'retrieve':SimpleNamespace(Retriever=lambda **kw: retr)}), \
                 patch.object(e6,'stdrag_hits',return_value={'q':hits}), \
                 patch.object(e6,'make_ours',side_effect=make_system), patch.object(e6,'make_llm',return_value=FakeLLM()):
                e6.main()
            files = list(preds.glob('e6_*.jsonl'))
            self.assertEqual(len(files),6)
            records = [json.loads(p.read_text(encoding='utf-8')) for p in files]
            self.assertTrue(all('error' not in r for r in records))
            self.assertTrue(all(r['gold4']=='EXCEEDS' and r['run_fingerprint'] for r in records))
            output = root/'report.md'
            with patch.object(sys,'argv',['report.py','--preds',str(preds),'--out',str(output)]):
                report.main()
            rendered = output.read_text(encoding='utf-8')
            self.assertIn('4범주 macro-F1',rendered)
            self.assertIn('모의 실행 포함',rendered)
            self.assertIn('stdrag → bm25rules',rendered)
            self.assertIn('gaterules → ours',rendered)

    def test_e5_missing_seeds_are_visible(self):
        with workspace_temp() as d:
            p = Path(d,'kpf_train_report_cb_kpf_s1.md')
            p.write_text('홀드아웃 — 정확도 0.95 · macro-F1 0.94\n',encoding='utf-8')
            output = Path(d,'e5')
            with patch.object(sys,'argv',['style_compare_report.py','--dir',d,'--out',str(output)]):
                style_compare_report.main()
            self.assertIn('학습 2,3',output.with_suffix('.md').read_text(encoding='utf-8'))

    def test_changed_fingerprint_rejects_resume(self):
        with workspace_temp() as d:
            p = Path(d,'pred.jsonl')
            p.write_text(json.dumps(protocol.stamp(record('a','NEI','NEI'),'old'))+'\n',encoding='utf-8')
            self.assertEqual(protocol.resume_ids(p,'old'), {'a'})
            with self.assertRaises(ValueError):
                protocol.resume_ids(p,'new')

    def test_pilot_exclusion_groups_claims(self):
        with workspace_temp() as d:
            p = Path(d,'pilot.jsonl')
            p.write_text(json.dumps({'id':'1__a','meta':{'claim_id':1}})+'\n',encoding='utf-8')
            items = [{'id':'1__b','meta':{'claim_id':1}}, {'id':'2__a','meta':{'claim_id':2}}]
            self.assertEqual([r['id'] for r in protocol.exclude_items(items,[p])], ['2__a'])

    def test_confirm_requires_independent_registered_gold(self):
        with workspace_temp() as d:
            p = Path(d,'claims.tsv')
            p.write_text('id\tclaim\texpected\ns1\tx\t범위 초과\n',encoding='utf-8')
            self.assertEqual(e6.load_claims(p)[0]['gold4'],'EXCEEDS')
            with self.assertRaises(ValueError):
                e6.load_claims(p,confirm=True)

    def test_historical_c15_exception_is_id_specific(self):
        with workspace_temp() as d:
            p = Path(d,'claims.tsv')
            p.write_text('id\tclaim\texpected\nC15\tx\t범위초과\nnew\ty\t범위초과\n',encoding='utf-8')
            self.assertEqual([r['gold4'] for r in e6.load_claims(p)],['NEI','EXCEEDS'])

    def test_report_old_error_normalized(self):
        with workspace_temp() as d:
            p = Path(d,'old.jsonl')
            p.write_text(json.dumps(record(1,'NEI','NEI',error='failure'))+'\n',encoding='utf-8')
            self.assertEqual(report.read_preds(p)[0]['pred'],'ERROR')

    def test_report_mixed_protocols_rejected(self):
        with workspace_temp() as d:
            p = Path(d,'pred.jsonl')
            p.write_text('\n'.join(json.dumps(record(i,'NEI','NEI',run_fingerprint=fp))
                                   for i,fp in [(1,'old'),(2,'new')]),encoding='utf-8')
            with self.assertRaises(ValueError):
                report.read_preds(p)

    def test_error_only_tables_do_not_crash_or_score_one(self):
        lines = []
        groups = {('ours','mock'):[record(1,'NEI','ERROR',error='x')]}
        report_safety.table_safety('scifact',groups,lines)
        report.table_verdict('scifact',groups,lines)
        self.assertIn('0/1 · 1','\n'.join(lines))

    def test_entity_errors_not_deliberate_abstentions(self):
        s = report_entity_tradeoff.summarize([{'noans':True,'hit1':False,'error':'bad'}])
        self.assertEqual((s['valid'],s['errors']), (0,1))
        self.assertIsNone(s['precision'])


class RetrievalTests(unittest.TestCase):
    def test_known_ir_metrics(self):
        self.assertEqual(ir.ndcg(['a','b'], {'a':2,'b':1}), 1)
        self.assertEqual(ir.recall(['a'], {'a':1,'b':1}), .5)
        self.assertEqual(ir.mrr(['b','a'], {'a':1}), .5)

    def test_candidate_selection_and_global_fallback(self):
        corpus = {str(i): {'title':'coffee','text':'risk'} for i in range(20)}
        mesh = {'0':{'mesh':[{'ui':'D1','name':'Coffee'}]}}
        idx = ir.Index(corpus,mesh,True)
        self.assertEqual(idx.candidates(['D1'])[0],[0])
        self.assertIsNone(idx.candidates(['missing'])[0])

    def test_e3_cli_counts_actual_visits(self):
        with workspace_temp() as d:
            root = Path(d); (root/'qrels').mkdir(); out = root/'out'
            corpus = [{' _unused':0,'_id':str(i),'title':'coffee','text':'risk'} for i in range(20)]
            (root/'corpus.jsonl').write_text('\n'.join(map(json.dumps,corpus)),encoding='utf-8')
            (root/'queries.jsonl').write_text(json.dumps({'_id':'q','text':'coffee risk'}),encoding='utf-8')
            (root/'qrels'/'test.tsv').write_text('query-id\tcorpus-id\tscore\nq\t0\t1\n',encoding='utf-8')
            (root/'mesh.json').write_text(json.dumps({'0':{'mesh':[{'ui':'D1','name':'Coffee'}]}}),encoding='utf-8')
            args = ['run_nfcorpus.py','--beir',d,'--mesh',str(root/'mesh.json'),'--systems','bm25_mesh_expand','gate','gate_rrf','--out',str(out)]
            extractor = lambda _: [{'ui':'D1','name':'Coffee'}]
            with patch.object(sys,'argv',args), patch.object(ir,'make_concept_extractor',return_value=extractor):
                ir.main()
            visited = {}
            for name in ('bm25_mesh_expand','gate','gate_rrf'):
                path = next(out.glob('ir_*_'+name+'_queries.jsonl'))
                visited[name] = json.loads(path.read_text(encoding='utf-8'))['visited']
            self.assertEqual(visited,{'bm25_mesh_expand':20,'gate':1,'gate_rrf':21})


if __name__ == '__main__':
    unittest.main()
