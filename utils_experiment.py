import logging
import os
import random
from typing import TypedDict, cast

import tenacity
from openai import AsyncOpenAI

from utils_openai import call_gpt4


def log_before_sleep(retry_state):
    logging.info(f"Retrying: attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep} seconds due to {retry_state.outcome.exception()}")



REORDER_PROMPT = """You can see several comics panels.
They were randomly shuffled.
Your goal is to recover the original order of the panels.

Format your output like this:
First, reason about what you see in each panel. Use all of them at once for relevant context. It should take not more than 1-2 short sentences per panel.
Then, reason about what you think the overall idea and plot of the comics is. Refer to panel numbers above. Try to put it into a narrative, explain how/why one panel goes after another. It can take 1-10 sententces (however, aim to be concise).
Then, your last line should contain the correct order of the panels (referring to panel numbers above, in the order you see them in). It has to always be formatted like this:
The correct order of the panels is: 5, 3, 1, 2, 4

Don't miss out any panels! You have to use each one!
"""


class ReorderResultDict(TypedDict):
    panels_path: str
    imgs_original: list[str]
    imgs_predicted: list[str]
    correct_order: list[int]
    predicted_order: list[int]
    reasoning: str
    failed: bool


class ReorderTaskContext:
    def __init__(self, len_imgs: int):
        self.imgs: list[str] = []
        self.correct_order: list[int | None] = [None] * len_imgs
        self.predicted_order: list[int] = []
        self.panels_path: str = ""


def handle_out_of_retries(retry_state: tenacity.RetryCallState):
    kwargs = retry_state.kwargs
    context = kwargs.get('context')
    imgs = context.imgs
    correct_order_set = set(context.correct_order)

    predicted_order = []
    predicted_set = set()
    for index in context.predicted_order:
        if index in predicted_set:
            continue
        if index not in correct_order_set:
            continue
        predicted_order.append(index)
        predicted_set.add(index)
    missing_indices = correct_order_set - predicted_set
    if missing_indices:
        predicted_order.extend(list(missing_indices))

    reasoning = "Default reasoning due to retries exhausted."
    return {
        "panels_path": context.panels_path,
        "imgs_original": imgs,
        "imgs_predicted": [imgs[i] for i in predicted_order],
        "correct_order": context.correct_order,
        "predicted_order": predicted_order,
        "reasoning": reasoning,
        "failed": True,
    }


async def run_reorder_task(
    client: AsyncOpenAI,
    panels_path: str,
    random_seed: int = 42,
    model_name: str = "gpt-4o-mini",
) -> ReorderResultDict:
    return await _run_reorder_task(
        client,
        panels_path,
        random_seed=random_seed,
        model_name=model_name,
        context=ReorderTaskContext(len(os.listdir(panels_path))),
    )


@tenacity.retry(
    wait=tenacity.wait_fixed(1),
    stop=tenacity.stop_after_attempt(1),  # TODO 5
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=log_before_sleep,
    retry_error_callback=handle_out_of_retries,
)
async def _run_reorder_task(
    client: AsyncOpenAI,
    panels_path: str,
    context: ReorderTaskContext,
    random_seed: int = 42,
    model_name: str = "gpt-4o-mini",
) -> ReorderResultDict:
    imgs = os.listdir(panels_path)
    context.imgs = imgs
    context.panels_path = panels_path
    random.seed(random_seed)
    random.shuffle(imgs)
    correct_order: list[int | None] = [None] * len(imgs)
    for shuffled_index, img in enumerate(imgs):
        actual_index = int(
            img.split(".")[0].replace("panel_", "")
        )
        correct_order[actual_index] = shuffled_index
        context.correct_order[actual_index] = shuffled_index

    response = await call_gpt4(
        client,
        REORDER_PROMPT,
        imgs_paths=[
            os.path.join(panels_path, img) for img in imgs
        ],
        model_name=model_name,
    )
    predicted_order = [
        int(num.strip().replace(".", "")) - 1
        for num in response.split(":")[-1].strip().split(",")
    ]
    context.predicted_order = predicted_order

    assert len(predicted_order) == len(imgs), "Incorrect number of panels in the response"
    reasoning = "\n".join(
        response.split(":")[:-1]
    )
    return {
        "panels_path": panels_path,
        "imgs_original": imgs,
        "imgs_predicted": [imgs[i] for i in predicted_order],
        "correct_order": cast(list[int], correct_order),
        "predicted_order": predicted_order,
        "reasoning": reasoning,
        "failed": False,
    }
