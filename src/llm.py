import ollama


MODEL_NAME = "gemma4:latest"


class LLM:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def generate(self, prompt):
        """
        Generate a response using the local Ollama model.
        """

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


if __name__ == "__main__":

    llm = LLM()

    prompt = """
    Answer the following question in one short sentence:

    What is the purpose of a vector database?
    """

    answer = llm.generate(prompt)

    print("\n" + "=" * 60)
    print("OLLAMA TEST")
    print("=" * 60)
    print(answer)
