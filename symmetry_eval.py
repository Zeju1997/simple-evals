"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
import os

import blobfile as bf
import pandas

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA, check_equality
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following yes-or-no question step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem and can only be "Yes" or "No".

{Question}

Remember to put your answer ("Yes" or "No") on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


class SymmetryEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: int | None = None, mode: str = "mnist"):
        csv_path = os.path.join(os.path.dirname(__file__), f"gb_{mode}_testset.csv")
        df = pandas.read_csv(
            # bf.BlobFile("https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv")
            csv_path
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)
        '''
        if self.api == "openai" or self.api == "claude" or self.api == "gemini" or self.api == "qwen":
            results = common.map_with_progress(fn, self.examples, 50)
        else:
            results = common.map_with_progress(fn, self.examples, 10)
        '''
        results = common.map_with_progress(fn, self.examples, 50)
        return common.aggregate_results(results)
