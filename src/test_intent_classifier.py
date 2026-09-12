from intent_classifier import IntentClassifier


TEST_CASES = [
    (
        "My iPhone battery is draining extremely fast.",
        "",
    ),
    (
        "I can't install the latest iOS update.",
        "",
    ),
    (
        "Bluetooth won't connect to my car.",
        "",
    ),
    (
        "My Apple ID password is locked and I cannot sign in.",
        "",
    ),
    (
        "How do I turn off notifications on my iPhone?",
        "",
    ),
]


def main() -> None:
    classifier = IntentClassifier()

    for customer_message, context in TEST_CASES:

        print()
        print("=" * 80)
        print("CUSTOMER:")
        print(customer_message)

        result = classifier.classify(
            customer_message=customer_message,
            context=context,
        )

        print("\nRESULT:")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reason:     {result['reason']}")


if __name__ == "__main__":
    main()