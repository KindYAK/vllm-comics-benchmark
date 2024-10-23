import base64
import logging

import tenacity
from openai import AsyncOpenAI

from semaphore import get_semaphore


def log_before_sleep(retry_state):
    logging.info(f"Retrying: attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep} seconds due to {retry_state.outcome.exception()}")


@tenacity.retry(
    # wait=tenacity.wait_fixed(30) + tenacity.wait_exponential(multiplier=1, min=30, max=60),
    # stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_fixed(1),
    stop=tenacity.stop_after_attempt(1),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=log_before_sleep
)
async def call_gpt4(
    client: AsyncOpenAI,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    model_name: str = "gpt-4o-mini",
    message_history: list[dict[str, str | list]] | None = None,
    imgs_paths: list[str] | None = None,
):
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": system_prompt
                },
            }
        )
    if message_history:
        messages.extend(message_history)
    messages.append({
        "role": "user",
        "content":
            [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
    })

    if imgs_paths:
        for img_path in imgs_paths:
            with open(img_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    }
                })

    async with get_semaphore("gpt-4"):
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
    response_txt = response.choices[0].message.content
    return response_txt
