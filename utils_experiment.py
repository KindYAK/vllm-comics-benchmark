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


@tenacity.retry(
    # wait=tenacity.wait_fixed(30) + tenacity.wait_exponential(multiplier=1, min=30, max=60),
    # stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_fixed(1),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=log_before_sleep
)
async def run_reorder_task(
    client: AsyncOpenAI,
    panels_path: str,
    random_seed: int = 42,
    model_name: str = "gpt-4o-mini",
) -> ReorderResultDict:
    imgs = os.listdir(panels_path)
    random.seed(random_seed)
    random.shuffle(imgs)
    correct_order: list[int | None] = [None] * len(imgs)
    for shuffled_index, img in enumerate(imgs):
        actual_index = int(
            img.split(".")[0].replace("panel_", "")
        )
        correct_order[actual_index] = shuffled_index

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
        "reasoning": reasoning
    }
