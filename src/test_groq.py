from llm_client import GroqClient


def main() -> None:

    client = GroqClient()

    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer-support "
                    "assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "Groq connection successful."
                ),
            },
        ],
        temperature=0.0,
    )

    print("Model:")
    print(client.model)

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()