import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

openai.api_key = "12345678"
ANTHROPIC_API_KEY = "12345678"


class OpenaiModel:
    def __init__(self, model_name="gpt-3.5-turbo", add_system_prompt=True) -> None:
        self.model_name = model_name
        self.add_system_prompt = add_system_prompt

    def fit_message(self, msg):
        if self.add_system_prompt:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ]
        else:
            conversation = [{"role": "user", "content": msg}]
        return conversation

    def __call__(self, msg, **kwargs):
        while True:
            try:
                raw_response = openai.ChatCompletion.create(
                    model=self.model_name, messages=self.fit_message(msg), **kwargs
                )
                self.raw_response = raw_response

                return [str(m.message.content) for m in raw_response["choices"]]
            except:
                pass

            time.sleep(10)


class AnthropicModel:
    def __init__(self, model_name="claude-2") -> None:
        self.model_name = model_name

        self.anthropic = Anthropic(
            api_key=ANTHROPIC_API_KEY,
        )

    def __call__(self, msg, **kwargs):
        while True:
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_name,
                    prompt=f"{HUMAN_PROMPT} {msg} {AI_PROMPT}",
                    **kwargs,
                )
                return completion.completion

            except:
                pass

            time.sleep(10)
