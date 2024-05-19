"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import blobfile as bf
import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

subject2category = {
    "animals": "other",
    "body": "other",
    "building": "other",
    "cloth": "other",
    "face": "other",
    "food_drinks_eatable": "other",
    "music_instruments": "other",
    "pen_writing": "other",
    "vehicle": "other"
}


class LOCALEval(Eval):
    def __init__(self, num_examples: int | None = None):
        df = pandas.read_csv(
            # bf.BlobFile("https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv")
            "/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/gb_testset.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=format_multichoice_question(row), role="user")
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(html=html, score=score, metrics={category: score}, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
