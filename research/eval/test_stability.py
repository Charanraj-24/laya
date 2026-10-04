"""Offline tests: python -m unittest research.eval.test_stability -v."""
from copy import deepcopy
import unittest

from laya.hooks import PredictContext

from research.eval import stability as s

QUESTIONS = {"intent": {
    "type": "choice", "instructions": "Choose the request category.",
    "criteria": {"billing": "payment issues", "technical": "software help", "sales": "new purchases"},
}}
DESCRIPTION_WEIGHTS = {"payment issues": 0.7, "software help": 0.2, "new purchases": 0.1}


def run(hook, states, questions, slot_scorer, collapse=()):
    """Mimic Agent.predict_batch: start hooks, one answer per question row, end hooks.

    `slot_scorer(state, keys_by_slot, texts_by_slot)` returns weights in slot order. As in Laya,
    slot s shows option `option_order[s]`, and probabilities come back in the question's own
    option order. Question ids in `collapse` report collapsed options in `usage["options"]`.
    """
    ctx = PredictContext(states=list(states), questions=questions)
    hook.on_predict_start(ctx)
    results = []
    for state in ctx.states:
        answers = {}
        for qid, q in ctx.questions.items():
            keys, texts = list(q["criteria"]), list(q["criteria"].values())
            order = q.get("option_order", list(range(len(keys))))
            weights = slot_scorer(state, [keys[i] for i in order], [texts[i] for i in order])
            total = sum(weights)
            probs = [0.0] * len(keys)
            for slot, option in enumerate(order):
                probs[option] = weights[slot] / total
            best = max(range(len(keys)), key=lambda i: (probs[i], -i))
            answers[qid] = {"type": "choice", "choice": keys[best],
                            "probabilities": dict(zip(keys, probs)),
                            "answer_confidence": probs[best], "low_confidence": False}
        usage = {"input_tokens": 10 * len(answers), "output_tokens": 0,
                 "truncated_questions": list(answers)}
        collapsed = {qid: {"total": 3, "distinct": 2, "tokens_per_option": 4}
                     for qid in answers if any(qid.startswith(c) for c in collapse)}
        if collapsed:
            usage["options"] = collapsed
        results.append({"model": "laya-rl-agent", "answers": answers, "usage": usage})
    ctx.results = results
    hook.on_predict_end(ctx)
    return ctx


def by_description(state, keys, texts):
    return [DESCRIPTION_WEIGHTS[t] for t in texts]


def by_first_slot(state, keys, texts):
    return [0.9] + [0.1 / (len(keys) - 1)] * (len(keys) - 1)


def by_label(state, keys, texts):
    weights = {"billing": 0.9, "technical": 0.05, "sales": 0.05, "A": 0.2, "B": 0.6, "C": 0.2}
    return [weights[k] for k in keys]


class VariantTests(unittest.TestCase):
    def test_variants_are_deterministic_unique_and_valid(self):
        hook = s.StabilityHook()
        first = hook.variants("intent", QUESTIONS["intent"])
        self.assertEqual(first, s.StabilityHook().variants("intent", QUESTIONS["intent"]))
        shapes = [(tuple(q.get("option_order", (0, 1, 2))), back is not None) for _, q, back in first]
        self.assertEqual(len(shapes), len(set(shapes)))
        self.assertNotIn(((0, 1, 2), False), shapes)  # the original is never repeated
        for _, q, back in first:
            self.assertEqual(sorted(q.get("option_order", [0, 1, 2])), [0, 1, 2])
            self.assertEqual(list(q["criteria"].values()), list(QUESTIONS["intent"]["criteria"].values()))
            if back:
                self.assertEqual(list(q["criteria"]), ["A", "B", "C"])
                self.assertEqual(back, {"A": "billing", "B": "technical", "C": "sales"})

    def test_two_options_and_rename_limit(self):
        two = {"type": "choice", "criteria": {"yes_please": "accept", "no_thanks": "decline"}}
        names = [name for name, _, _ in s.StabilityHook(shuffles=5).variants("q", two)]
        self.assertEqual(len(names), 3)  # reversed, rename, rename_reversed: nothing else is distinct
        many = {"type": "choice", "criteria": {"k%d" % i: "option %d" % i for i in range(25)}}
        names = [name for name, _, _ in s.StabilityHook().variants("q", many)]
        self.assertFalse(any(n.startswith("rename") for n in names))
        names = [name for name, _, _ in s.StabilityHook(renames=False).variants("intent", QUESTIONS["intent"])]
        self.assertFalse(any(n.startswith("rename") for n in names))

    def test_own_option_order_is_not_copied(self):
        question = dict(QUESTIONS["intent"], option_order=[2, 0, 1])
        copies = {name: q for name, q, _ in s.StabilityHook().variants("intent", question)}
        self.assertNotIn("option_order", copies["rename"])  # shown in the canonical order
        self.assertEqual(copies["reversed"]["option_order"], [2, 1, 0])


class HookTests(unittest.TestCase):
    def test_stable_answers_and_clean_output(self):
        questions = deepcopy(QUESTIONS)
        ctx = run(s.StabilityHook(), ["a", "b"], questions, by_description, collapse=("intent::stab",))
        self.assertIs(ctx.questions, questions)
        self.assertEqual(questions, QUESTIONS)
        for result in ctx.results:
            self.assertEqual(list(result["answers"]), ["intent"])
            answer = result["answers"]["intent"]
            self.assertEqual(answer["choice"], "billing")
            rel = answer["reliability"]
            self.assertEqual(rel["stability"], 1.0)
            self.assertAlmostEqual(rel["soft_stability"], 0.7)
            self.assertEqual(rel["distinct_choices"], ["billing"])
            self.assertEqual(rel["n_variants"], len(rel["variant_choices"]) + 1)
            self.assertEqual(rel["variants_collapsed"], rel["n_variants"] - 1)
            self.assertEqual(result["usage"]["truncated_questions"], ["intent"])
            self.assertNotIn("options", result["usage"])

    def test_position_sensitive_answers_are_flagged(self):
        rel = run(s.StabilityHook(), ["a"], deepcopy(QUESTIONS), by_first_slot).results[0]["answers"]["intent"]["reliability"]
        self.assertLess(rel["stability"], 1.0)
        self.assertLess(rel["soft_stability"], 0.9)
        self.assertGreater(len(rel["distinct_choices"]), 1)

    def test_label_sensitive_answers_flip_only_under_rename(self):
        rel = run(s.StabilityHook(), ["a"], deepcopy(QUESTIONS), by_label).results[0]["answers"]["intent"]["reliability"]
        for name, choice in rel["variant_choices"].items():
            # rename answers come back under the caller's keys, never as A/B/C
            self.assertEqual(choice, "technical" if name.startswith("rename") else "billing")

    def test_same_copies_for_every_state(self):
        ctx = run(s.StabilityHook(), ["a", "b", "c"], deepcopy(QUESTIONS), by_first_slot)
        rels = [r["answers"]["intent"]["reliability"] for r in ctx.results]
        self.assertTrue(all(r == rels[0] for r in rels))

    def test_other_questions_untouched(self):
        questions = deepcopy(QUESTIONS)
        questions["urgent"] = {"type": "noul", "instructions": "Urgent?"}
        questions["single"] = {"type": "choice", "criteria": {"only": "the one option"}}
        hook = s.StabilityHook()
        ctx = PredictContext(states=["a"], questions=questions)
        hook.on_predict_start(ctx)
        self.assertNotIn("urgent::stab1", ctx.questions)
        self.assertNotIn("single::stab1", ctx.questions)
        self.assertIn("intent::stab1", ctx.questions)

    def test_failed_or_skipped_calls(self):
        hook = s.StabilityHook()
        questions = deepcopy(QUESTIONS)
        ctx = PredictContext(states=["a"], questions=questions)
        hook.on_predict_start(ctx)
        hook.on_predict_end(ctx)  # inference failed: no results
        self.assertIs(ctx.questions, questions)
        skipped = PredictContext(states=["a"], questions=questions)
        skipped.skip([{"answers": {}}])
        hook.on_predict_start(skipped)
        self.assertIs(skipped.questions, questions)
        hook.on_predict_end(skipped)

    def test_reserved_separator(self):
        ctx = PredictContext(states=["a"], questions={"bad::stab1": QUESTIONS["intent"]})
        with self.assertRaises(ValueError):
            s.StabilityHook().on_predict_start(ctx)
        with self.assertRaises(ValueError):
            s.StabilityHook(shuffles=-1)


class MetricTests(unittest.TestCase):
    def test_signal_metrics_and_bootstrap(self):
        scores, correct = [0.9, 0.8, 0.3, 0.2], [True, True, False, False]
        metrics = s.signal_metrics(scores, correct)
        self.assertEqual(metrics["auroc"], 1)
        self.assertEqual(metrics["selective_accuracy@50"], 1)
        self.assertEqual(metrics["distinct_values"], 4)
        rows = [{"good": g, "bad": 1 - g, "correct": c} for g, c in zip(scores, correct)]
        diff = s.bootstrap_difference(rows, "good", "bad", "auroc", n_boot=200)
        self.assertGreater(diff["low"], 0)
        self.assertIsNone(s.bootstrap_difference([{"good": 1, "bad": 0, "correct": True}],
                                                 "good", "bad", "auroc", n_boot=20))


if __name__ == "__main__":
    unittest.main()
