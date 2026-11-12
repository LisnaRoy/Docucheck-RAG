import ollama

MODEL = "llama3.2:3b"


def generate(prompt: str, system: str | None = None, model: str = MODEL) -> str:
    """Send a prompt to the local Ollama model and return the text reply."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0},
    )
    return response["message"]["content"]