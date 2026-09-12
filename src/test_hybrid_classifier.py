from hybrid_intent_classifier import HybridIntentClassifier


def main() -> None:

    classifier = HybridIntentClassifier(
        llm_classifier=None
    )

    tests = [
        (
            "My iPhone battery is draining very quickly."
        ),
        (
            "Bluetooth won't connect to my car."
        ),
        (
            "My Apple ID password is locked."
        ),
        (
            "I can't install the latest iOS update."
        ),
        (
            "How do I turn off notifications?"
        ),
        (
            "My iPhone keeps freezing and restarting."
        ),
        (
            "My screen is cracked."
        ),
        (
            "I need a repair under warranty."
        ),
    ]

    for message in tests:

        result = classifier.classify(
            customer_message=message
        )

        print()
        print(message)
        print(
            f" -> {result['intent']} "
            f"confidence={result['confidence']:.2f} "
            f"method={result['method']}"
        )


if __name__ == "__main__":
    main()