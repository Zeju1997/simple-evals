import base64
import time
from typing import Any

import openai
from openai import OpenAI, AzureOpenAI

from ..types import MessageList, SamplerBase

from dashscope import Generation
import dashscope
dashscope.api_key = "sk-6bfca444643a4aef899f5e5426173872"

QWEN_SYSTEM_MESSAGE_API = "You are a helpful assistant."

class QwenCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "qwen-turbo",
        system_message: str | None = None,
        temperature: float = 0, # 0.5
        max_tokens: int = 2000,
    ):
        self.api_key_name = "DASHSCOPE_API_KEY"
        self.client = Generation()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        if model == "qwen-turbo":
            self.model = Generation.Models.qwen_turbo
        elif model == "qwen-plus":
            self.model = Generation.Models.qwen_plus
        else:
            self.model = Generation.Models.qwen_max
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.call(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    # max_tokens=self.max_tokens,
                    result_format='message',
                )
                print(response)
                return response.output.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
