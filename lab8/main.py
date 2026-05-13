import requests
from typing import Literal
from time import time
Reasoning = Literal["off", "low", "medium", "high", "on"]
QWEN = "qwen/qwen3.5-9b"
NEMOTRON = "nvidia/nemotron-3-nano-4b"
URL = "http://127.0.0.1:1234/api/v1/chat"

with open("ENG_article.txt", 'r', encoding='cp1251') as file:
    text = file.read().replace('\n', ' ')


def generate_prompt(input_prompt: str, system_prompt: str, temperature: float = 0.3,
                    model: str = "nvidia/nemotron-3-nano-4b", max_tokens: int = 1024,
                    reasoning: Reasoning = "off") -> dict:
    """
    Generate prompt
        :param str input_prompt: Message to send to the model.
        :param str system_prompt: System message that sets model behavior or instructions.
        :param float temperature: Randomness in token selection. 0 is deterministic,
                                  higher values increase creativity [0,1].
        :param str model: Unique identifier for the model to use.
        :param int max_tokens: Maximum number of tokens to generate.
        :param Reasoning reasoning: Reasoning setting. Will error if the model being used does not support the reasoning
                                    setting using. Defaults to the automatically chosen setting for the model.
                                    Allowed values: "off" | "low" | "medium" | "high" | "on"
        :return: prompt json format
        :rtype dict:
    """
    return {
        "model": model,
        "input": input_prompt,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "reasoning": reasoning,
        "max_output_tokens": max_tokens,
    }


def send_message(url: str, data: dict) -> dict:
    return requests.post(url, json=data).json()


questions = [
    "В каком году была обозначена проблема взрывающихся градиентов?",
    "Кто в 1891 году разработал метод уничтожающей производной?",
    "Кто предложил цепное правило дифференцирования и в каком году?",
]
sys_prompt = "Используй только переданный текст. Отвечай на русском языке."
with open("results1.txt", "w", encoding='utf-8-sig') as f:
    for i, q in enumerate(questions):
        start = time()
        r = send_message(URL,
                         generate_prompt(
                             input_prompt=text,
                             system_prompt=f"{q} {sys_prompt}",
                             # model=QWEN,
                             # max_tokens=100,
                             # reasoning="high",
                             reasoning="off",
                         )).get("output")[0].get("content")
        print(f"Time taken: {time() - start:.2f} c")
        print(r)
        f.write(f"{i+1}. {r}\n\n")
