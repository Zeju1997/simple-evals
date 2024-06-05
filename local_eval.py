"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
import os

import blobfile as bf
import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

'''
subject2category = {
    "aerial_crafts": "transportation",
    "land_crafts": "transportation",
    "entertainment": "sociaty",
    "science": "sociaty",
    "building": "sociaty",
    "human": "living_being",
    "animal": "living_being",
    "beverage": "food_drinks",
    "food": "food_drinks",
    "dairy": "food_drinks",
    "fruit": "food_drinks",
    "accessory": "daily_objects",
    "computer": "daily_objects",
    "book": "daily_objects",
    "furniture": "daily_objects",
    "musical_instrument": "daily_objects",
    "tool": "daily_objects",
    "clothing": "daily_objects",
    "time_clock": "daily_objects",
}
'''

subject2category = {
    "aerial_crafts": "other",
    "land_crafts": "other",
    "entertainment": "other",
    "science": "other",
    "building": "other",
    "human": "other",
    "animal": "other",
    "beverage": "other",
    "food": "other",
    "dairy": "other",
    "fruit": "other",
    "accessory": "other",
    "computer": "other",
    "book": "other",
    "furniture": "other",
    "musical_instrument": "other",
    "tool": "other",
    "clothing": "other",
    "time_clock": "other",
}

class LOCALEval(Eval):
    def __init__(self, num_examples: int | None = None, mode: str = "raw"):
        csv_path = os.path.join(os.path.dirname(__file__), "gb_{}_testset.csv".format(mode))
        df = pandas.read_csv(
            # bf.BlobFile("https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv")
            csv_path
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            max_retries = 5
            retry_count = 0
            extracted_answer = None
            response_text = ""
            prompt_messages = [
                sampler._pack_message(content=format_multichoice_question(row), role="user")
            ]

            while retry_count < max_retries and extracted_answer is None:
                response_text = sampler(prompt_messages)
                match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
                extracted_answer = match.group(1) if match else None
                retry_count += 1

            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            # category = subject2category.get(row["Subject"], "other")
            try:
                category = subject2category[row["Subject"]]
            except KeyError:
                # Handle the case where the subject is not found
                raise KeyError(f"Subject '{row['Subject']}' not found in subject2category.")
            return SingleEvalResult(html=html, score=score, metrics={category: score}, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
