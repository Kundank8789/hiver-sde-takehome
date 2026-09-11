from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "apple_support_interactions.jsonl"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "blind_golden_candidates.jsonl"
)

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "blind_golden_annotation.csv"
)

SEED = "hiver-apple-support-v1"

TOTAL_EXAMPLES = 200
EXAMPLES_PER_BUCKET = 20

# Keep a small deterministic candidate reservoir per bucket.
RESERVOIR_SIZE = 40

# Fallback candidates used if overlaps prevent us
# from reaching exactly 200 unique examples.
FALLBACK_RESERVOIR_SIZE = 500


# ============================================================
# Topic buckets
#
# IMPORTANT:
# These buckets are used ONLY for sampling coverage.
# They are NEVER written into the blind annotation file.
# ============================================================

BUCKET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "IOS_UPDATE": (
        "ios update",
        "software update",
        "update",
        "updating",
        "upgrade",
        "upgrading",
        "install update",
        "can't update",
        "cannot update",
        "update error",
        "update stuck",
        "macos update",
    ),

    "BATTERY_POWER": (
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "draining",
        "charging",
        "charge",
        "charger",
        "power",
        "dies",
    ),

    "PERFORMANCE_STABILITY": (
        "slow",
        "slower",
        "lag",
        "lagging",
        "freeze",
        "freezing",
        "crash",
        "crashing",
        "restart",
        "restarting",
        "reboot",
        "stuck",
        "hanging",
        "hang",
        "unresponsive",
    ),

    "CONNECTIVITY": (
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "network",
        "connection",
        "connect",
        "connected",
        "no service",
        "hotspot",
    ),

    "APPS_MEDIA": (
        "app",
        "apps",
        "apple music",
        "music",
        "itunes",
        "imovie",
        "facetime",
        "messages",
        "message",
        "mail",
        "photos",
        "photo",
        "youtube",
    ),

    "DEVICE_HARDWARE": (
        "screen",
        "display",
        "speaker",
        "microphone",
        "keyboard",
        "key",
        "camera",
        "charger port",
        "broken",
        "crack",
        "cracked",
        "damaged",
        "hardware",
        "won't turn on",
        "doesn't turn on",
    ),

    "FEATURE_HOW_TO": (
        "how do i",
        "how can i",
        "how to",
        "where can i",
        "where is",
        "how does",
        "can i use",
        "is there a way",
    ),

    "ACCOUNT_SECURITY": (
        "apple id",
        "password",
        "account",
        "verification",
        "verify",
        "locked",
        "disabled",
        "sign in",
        "login",
        "security",
        "icloud password",
    ),

    "PURCHASE_REPAIR_WARRANTY": (
        "order",
        "ordered",
        "purchase",
        "purchased",
        "buy",
        "bought",
        "refund",
        "warranty",
        "repair",
        "appointment",
        "reservation",
        "store",
        "replacement",
        "service",
    ),
}


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Candidate:
    interaction_id: int
    item: dict[str, Any]
    rank: str


# ============================================================
# Helpers
# ============================================================

def stable_rank(interaction_id: int, bucket: str = "global") -> str:
    """
    Produce a deterministic ranking.

    Same interaction + same seed + same bucket
    always produces the same value.
    """
    payload = f"{SEED}:{bucket}:{interaction_id}".encode(
        "utf-8"
    )

    return hashlib.sha256(payload).hexdigest()


def add_to_reservoir(
    reservoir: list[Candidate],
    candidate: Candidate,
    limit: int,
) -> None:
    """
    Keep the lowest-ranked candidates.

    This gives us deterministic sampling without storing
    the entire dataset in memory.
    """
    reservoir.append(candidate)

    if len(reservoir) > limit:
        reservoir.sort(key=lambda x: x.rank)
        del reservoir[limit:]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def get_bucket_matches(text: str) -> list[str]:
    """
    Determine which sampling buckets an interaction
    could contribute to.

    These are NOT labels.
    """
    matches: list[str] = []

    for bucket, keywords in BUCKET_KEYWORDS.items():

        if any(
            keyword in text
            for keyword in keywords
        ):
            matches.append(bucket)

    return matches


def build_csv_row(
    item: dict[str, Any],
) -> dict[str, str]:

    context_messages = item.get(
        "context",
        [],
    )

    context_text = "\n".join(
        f"[{message.get('role', '').upper()}] "
        f"{message.get('text', '')}"
        for message in context_messages
        if message.get("tweet_id")
        != item.get("customer_tweet_id")
    )

    return {
        "interaction_id": str(
            item["interaction_id"]
        ),

        "customer_tweet_id": str(
            item["customer_tweet_id"]
        ),

        "agent_tweet_id": str(
            item["agent_tweet_id"]
        ),

        "customer_message": str(
            item["customer_message"]
        ),

        "context": context_text,

        "historical_agent_response": str(
            item["agent_response"]
        ),

        # HUMAN ANNOTATION FIELDS
        "human_intent": "",
        "should_escalate": "",
        "escalation_reason": "",
    }


# ============================================================
# Main preparation pipeline
# ============================================================

def main() -> None:

    print("=" * 72)
    print("HIVER — BLIND GOLDEN SET PREPARATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            f"Make sure this file exists before running "
            f"the preparation pipeline."
        )

    print(f"\nInput: {INPUT_PATH}")
    print(f"Target examples: {TOTAL_EXAMPLES}")
    print(
        f"Examples per topic bucket: "
        f"{EXAMPLES_PER_BUCKET}"
    )

    # --------------------------------------------------------
    # Initialize reservoirs
    # --------------------------------------------------------

    reservoirs: dict[str, list[Candidate]] = {
        bucket: []
        for bucket in BUCKET_KEYWORDS
    }

    # Messages with no obvious keyword signal.
    reservoirs["OTHER_UNCLEAR"] = []

    fallback_reservoir: list[Candidate] = []

    total_read = 0
    valid_read = 0

    # --------------------------------------------------------
    # Single streaming pass over JSONL
    # --------------------------------------------------------

    print("\nScanning support interactions...")

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            total_read += 1

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed JSON "
                    f"at line {line_number}: {exc}"
                )
                continue

            required_fields = {
                "interaction_id",
                "customer_tweet_id",
                "agent_tweet_id",
                "customer_message",
                "context",
                "agent_response",
            }

            if not required_fields.issubset(item):
                continue

            valid_read += 1

            interaction_id = int(
                item["interaction_id"]
            )

            text = normalize_text(
                item["customer_message"]
            )

            matched_buckets = get_bucket_matches(
                text
            )

            if not matched_buckets:

                candidate = Candidate(
                    interaction_id=interaction_id,
                    item=item,
                    rank=stable_rank(
                        interaction_id,
                        "OTHER_UNCLEAR",
                    ),
                )

                add_to_reservoir(
                    reservoirs["OTHER_UNCLEAR"],
                    candidate,
                    RESERVOIR_SIZE,
                )

            else:

                for bucket in matched_buckets:

                    candidate = Candidate(
                        interaction_id=interaction_id,
                        item=item,
                        rank=stable_rank(
                            interaction_id,
                            bucket,
                        ),
                    )

                    add_to_reservoir(
                        reservoirs[bucket],
                        candidate,
                        RESERVOIR_SIZE,
                    )

            # Global fallback reservoir.
            fallback_candidate = Candidate(
                interaction_id=interaction_id,
                item=item,
                rank=stable_rank(
                    interaction_id,
                    "GLOBAL",
                ),
            )

            add_to_reservoir(
                fallback_reservoir,
                fallback_candidate,
                FALLBACK_RESERVOIR_SIZE,
            )

    print(
        f"\nInteractions read: {total_read:,}"
    )

    print(
        f"Valid interactions: {valid_read:,}"
    )

    if valid_read == 0:
        raise RuntimeError(
            "No valid support interactions were found."
        )

    # --------------------------------------------------------
    # Select exactly 20 unique examples per bucket.
    # --------------------------------------------------------

    selected: list[Candidate] = []
    selected_ids: set[int] = set()

    bucket_order = [
        *BUCKET_KEYWORDS.keys(),
        "OTHER_UNCLEAR",
    ]

    print("\nSampling coverage:")

    for bucket in bucket_order:

        candidates = sorted(
            reservoirs[bucket],
            key=lambda candidate: candidate.rank,
        )

        bucket_selected = 0

        for candidate in candidates:

            if (
                candidate.interaction_id
                in selected_ids
            ):
                continue

            selected.append(candidate)
            selected_ids.add(
                candidate.interaction_id
            )

            bucket_selected += 1

            if (
                bucket_selected
                >= EXAMPLES_PER_BUCKET
            ):
                break

        print(
            f"  {bucket:<28} "
            f"{bucket_selected:>3}"
        )

    # --------------------------------------------------------
    # Fill to exactly 200 if overlaps reduced the count.
    # --------------------------------------------------------

    if len(selected) < TOTAL_EXAMPLES:

        print(
            "\nFilling remaining slots "
            "from global deterministic reservoir..."
        )

        for candidate in sorted(
            fallback_reservoir,
            key=lambda candidate: candidate.rank,
        ):

            if (
                candidate.interaction_id
                in selected_ids
            ):
                continue

            selected.append(candidate)
            selected_ids.add(
                candidate.interaction_id
            )

            if len(selected) >= TOTAL_EXAMPLES:
                break

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if len(selected) != TOTAL_EXAMPLES:
        raise RuntimeError(
            f"Could only select "
            f"{len(selected)} unique examples; "
            f"expected {TOTAL_EXAMPLES}."
        )

    # Deterministic final order.
    selected.sort(
        key=lambda candidate: stable_rank(
            candidate.interaction_id,
            "FINAL_ORDER",
        )
    )

    # --------------------------------------------------------
    # Write blind JSONL
    # --------------------------------------------------------

    CANDIDATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CANDIDATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for candidate in selected:

            clean_item = {
                "interaction_id": candidate.item[
                    "interaction_id"
                ],

                "customer_tweet_id": candidate.item[
                    "customer_tweet_id"
                ],

                "agent_tweet_id": candidate.item[
                    "agent_tweet_id"
                ],

                "customer_message": candidate.item[
                    "customer_message"
                ],

                "context": candidate.item[
                    "context"
                ],

                "agent_response": candidate.item[
                    "agent_response"
                ],
            }

            file.write(
                json.dumps(
                    clean_item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Write blind CSV for Zoho/Excel
    # --------------------------------------------------------

    fieldnames = [
        "interaction_id",
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_message",
        "context",
        "historical_agent_response",
        "human_intent",
        "should_escalate",
        "escalation_reason",
    ]

    with ANNOTATION_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for candidate in selected:

            writer.writerow(
                build_csv_row(candidate.item)
            )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    with ANNOTATION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    if len(rows) != TOTAL_EXAMPLES:
        raise RuntimeError(
            f"CSV verification failed: "
            f"found {len(rows)} rows."
        )

    expected_fields = set(fieldnames)

    actual_fields = set(
        reader.fieldnames or []
    )

    if actual_fields != expected_fields:
        raise RuntimeError(
            "CSV schema verification failed.\n"
            f"Expected: {sorted(expected_fields)}\n"
            f"Actual:   {sorted(actual_fields)}"
        )

    # Ensure blind annotation fields are empty.
    for index, row in enumerate(
        rows,
        start=2,
    ):
        if (
            row["human_intent"]
            or row["should_escalate"]
            or row["escalation_reason"]
        ):
            raise RuntimeError(
                f"Blindness check failed at CSV row "
                f"{index}."
            )

    print()
    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)

    print(
        f"Unique blind examples: "
        f"{len(rows)}"
    )

    print(
        f"Candidate JSONL: "
        f"{CANDIDATE_PATH}"
    )

    print(
        f"Annotation CSV: "
        f"{ANNOTATION_PATH}"
    )

    print(
        "\nThe annotation CSV contains NO "
        "provisional labels."
    )


if __name__ == "__main__":
    main()