"""
=====================================================
Railway AI Assistant — Synthetic Dataset Generator
=====================================================
Uses Google Gemini API to generate 1000+ high-quality
multi-turn railway conversations for fine-tuning T5.

Usage:
    python generate_dataset.py

Output:
    data/dataset.csv         → Full dataset
    data/dataset_sample.csv  → First 50 rows (for review)
"""

import os
import json
import time
import random
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# ── Load API Key ────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found!\n"
        "Create a .env file and add: GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ── Railway Domain Knowledge ─────────────────────────────────────────────────
DESTINATIONS = [
    "Manchester", "Leeds", "Edinburgh", "Cardiff", "York",
    "Birmingham", "Bristol", "Liverpool", "Newcastle", "Sheffield",
    "Nottingham", "Leicester", "Exeter", "Brighton", "Oxford",
    "Cambridge", "Glasgow", "Gatwick", "Heathrow", "Wembley",
    "Lime Street", "Paddington", "Victoria", "Waterloo", "King's Cross"
]

CONSTRAINTS = [
    "bike", "bicycle", "pet", "dog", "cat",
    "wheelchair", "accessibility needs", "large luggage",
    "pram", "pushchair", "baby buggy"
]

TIME_CONTEXTS = [
    "tomorrow morning", "tonight", "urgently", "before midnight",
    "first train", "last train", "next available", "peak hours",
    "off-peak", "early morning", "this evening", "this afternoon"
]

PRIORITIES = [
    "cheapest", "fastest", "fewest changes", "direct only",
    "most comfortable", "quickest route"
]

DISRUPTION_QUERIES = [
    "Are there any delays?",
    "Show current disruptions",
    "Is my train running on time?",
    "Any cancellations today?",
    "What's causing the delay?"
]

FOLLOW_UP_TYPES = [
    "adding a constraint",
    "changing the time",
    "asking for platform info",
    "asking about ticket price",
    "asking about journey duration",
    "requesting fewer changes",
    "checking for disruptions",
    "asking about return journey",
    "correcting the destination",
    "asking about accessibility"
]

# ── Scenario Templates ───────────────────────────────────────────────────────
SCENARIO_TEMPLATES = [
    # Template 1: Simple booking with constraint added mid-flow
    {
        "type": "constraint_added",
        "description": "User searches for train, then adds a constraint",
        "turns": 2
    },
    # Template 2: Urgent journey
    {
        "type": "urgent_journey",
        "description": "User needs to reach destination urgently or before deadline",
        "turns": 2
    },
    # Template 3: Multi-turn with correction
    {
        "type": "correction_flow",
        "description": "User corrects destination or time mid-conversation",
        "turns": 3
    },
    # Template 4: Disruption handling
    {
        "type": "disruption_query",
        "description": "User asks about delays and requests alternatives",
        "turns": 2
    },
    # Template 5: Complex multi-constraint
    {
        "type": "multi_constraint",
        "description": "User has multiple constraints like bike + accessibility",
        "turns": 3
    },
    # Template 6: Return journey planning
    {
        "type": "return_journey",
        "description": "User plans outward then return journey",
        "turns": 3
    },
    # Template 7: Platform and live info
    {
        "type": "live_info",
        "description": "User asks for live platform/departure info",
        "turns": 2
    },
    # Template 8: Price-sensitive journey
    {
        "type": "price_sensitive",
        "description": "User wants cheapest available option",
        "turns": 2
    }
]


# ── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(scenario: dict, destination: str, constraint: str,
                 time_ctx: str, priority: str, follow_up: str) -> str:
    """
    Build a detailed prompt for Gemini to generate one conversation example.
    Returns a JSON string with conversation_history and target_response.
    """

    prompt = f"""
You are generating training data for a Railway AI Conversational Assistant.

Generate a realistic multi-turn conversation between a USER and a RAILWAY ASSISTANT.

Scenario type: {scenario['type']}
Description: {scenario['description']}
Number of turns: {scenario['turns']}

Use these specific details:
- Destination: {destination}
- Passenger constraint: {constraint}
- Time context: {time_ctx}
- Journey priority: {priority}
- Follow-up type: {follow_up}

IMPORTANT RULES:
1. The conversation must feel natural, like a real passenger using a chatbot
2. The assistant must REMEMBER context from earlier turns (e.g., destination already mentioned)
3. The final assistant response must include a structured JSON state embedded inside it
4. Handle mid-flow corrections gracefully if applicable
5. Keep user messages short and realistic (how people actually type)

Return ONLY a valid JSON object in this exact format (no extra text, no markdown):
{{
  "scenario_type": "{scenario['type']}",
  "conversation_history": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "final_state": {{
    "origin": "London",
    "destination": "{destination}",
    "date": "...",
    "time": "...",
    "journey_type": "direct or with_changes",
    "constraints": ["{constraint}"],
    "priority": "{priority}",
    "status": "resolved or needs_info"
  }},
  "input_text": "Summarized conversation history as a single input string for the model",
  "target_response": "The ideal final assistant response text"
}}
"""
    return prompt


# ── Single Example Generator ─────────────────────────────────────────────────
def generate_single_example(scenario: dict) -> dict | None:
    """
    Call Gemini API to generate one conversation example.
    Returns parsed dict or None if failed.
    """
    destination = random.choice(DESTINATIONS)
    constraint  = random.choice(CONSTRAINTS)
    time_ctx    = random.choice(TIME_CONTEXTS)
    priority    = random.choice(PRIORITIES)
    follow_up   = random.choice(FOLLOW_UP_TYPES)

    prompt = build_prompt(scenario, destination, constraint,
                          time_ctx, priority, follow_up)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,        # High creativity for diverse data
                max_output_tokens=1024,
            )
        )

        raw_text = response.text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        return parsed

    except json.JSONDecodeError as e:
        print(f"\n⚠️  JSON parse error: {e} — skipping this example")
        return None
    except Exception as e:
        print(f"\n⚠️  API error: {e} — retrying after 5s")
        time.sleep(5)
        return None


# ── Rule-Based Fallback Generator ────────────────────────────────────────────
def generate_rule_based_examples(count: int = 200) -> list[dict]:
    """
    Generate simple rule-based examples as fallback/padding.
    These are simpler but ensure we always hit our target count.
    """
    examples = []

    simple_templates = [
        # (user_msg_template, assistant_response_template)
        (
            "I need a {priority} train to {dest} {time}",
            "Sure! Searching for {priority} trains to {dest} {time}. "
            "Found 3 options. The earliest departs at 07:15 from Platform 3."
        ),
        (
            "Train to {dest} please",
            "Happy to help! When would you like to travel to {dest}?"
        ),
        (
            "What's the next train to {dest}?",
            "The next train to {dest} departs in 12 minutes from Platform 2. "
            "It arrives at {dest} at 14:35."
        ),
        (
            "I need to get to {dest} {time}. I also have a {constraint}.",
            "Got it! Looking for trains to {dest} {time} with {constraint} space. "
            "Please note that {constraint} reservations are required on this route."
        ),
        (
            "Show me trains to {dest} with {constraint} space",
            "Here are trains to {dest} with {constraint} accommodation: "
            "• 08:30 - Platform 4 (2 {constraint} spaces left) "
            "• 10:15 - Platform 1 (spaces available) "
            "• 12:00 - Platform 6 (fully booked)"
        ),
        (
            "Any disruptions to {dest} today?",
            "There are currently minor delays on the {dest} line due to engineering works. "
            "Trains are running approximately 15 minutes late. "
            "I recommend allowing extra time for your journey."
        ),
        (
            "Cheapest way to get to {dest}?",
            "The cheapest option to {dest} is an Advance ticket at £{price}. "
            "Available on the 06:45 and 11:30 departures. "
            "Off-peak tickets from £{price2} are available after 09:30."
        ),
        (
            "I need to reach {dest} {time}. My flight lands in 2 hours.",
            "For an urgent journey to {dest}, the fastest option departs in 8 minutes. "
            "Journey time: 45 minutes. You'll arrive with time to spare. "
            "Platform 5 — please go now!"
        ),
    ]

    for _ in range(count):
        dest        = random.choice(DESTINATIONS)
        constraint  = random.choice(CONSTRAINTS)
        time_ctx    = random.choice(TIME_CONTEXTS)
        priority    = random.choice(PRIORITIES)
        price       = random.randint(15, 80)
        price2      = price + random.randint(5, 20)

        template = random.choice(simple_templates)
        user_msg  = template[0].format(
            dest=dest, constraint=constraint,
            time=time_ctx, priority=priority
        )
        asst_msg  = template[1].format(
            dest=dest, constraint=constraint,
            time=time_ctx, priority=priority,
            price=price, price2=price2
        )

        examples.append({
            "scenario_type": "rule_based",
            "input_text": f"User: {user_msg}",
            "target_response": asst_msg,
            "final_state": json.dumps({
                "destination": dest,
                "constraints": [constraint],
                "time": time_ctx,
                "priority": priority,
                "status": "resolved"
            })
        })

    return examples


# ── Flatten for CSV ──────────────────────────────────────────────────────────
def flatten_example(example: dict) -> dict:
    """
    Convert a generated example into a flat CSV row.
    """
    # Build conversation history string
    history_str = ""
    if "conversation_history" in example:
        turns = example["conversation_history"]
        history_str = "\n".join(
            [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        )
    else:
        history_str = example.get("input_text", "")

    return {
        "scenario_type"    : example.get("scenario_type", "unknown"),
        "conversation_history": history_str,
        "input_text"       : example.get("input_text", history_str),
        "target_response"  : example.get("target_response", ""),
        "final_state"      : json.dumps(example.get("final_state", {}))
    }


# ── Main Generator ───────────────────────────────────────────────────────────
def generate_dataset(target_count: int = 1000):
    """
    Main function: generates full dataset using Gemini API + rule-based fallback.

    Strategy:
    - ~800 examples from Gemini (high quality, diverse)
    - ~200 examples rule-based (fast, as padding)
    - Total: 1000+ rows
    """

    os.makedirs("data", exist_ok=True)

    all_examples   = []
    gemini_target  = 800
    failed_count   = 0
    max_fails      = 50   # Stop if too many consecutive failures

    print("=" * 60)
    print("  🚂 Railway AI — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Target: {target_count} examples")
    print(f"  Gemini API: {gemini_target} examples")
    print(f"  Rule-based: {target_count - gemini_target} examples")
    print("=" * 60)

    # ── Phase 1: Gemini API Examples ──────────────────────────────────────
    print("\n📡 Phase 1: Generating Gemini API examples...\n")

    with tqdm(total=gemini_target, desc="Generating", unit="example") as pbar:
        while len(all_examples) < gemini_target and failed_count < max_fails:

            # Rotate through all scenario types evenly
            scenario = SCENARIO_TEMPLATES[len(all_examples) % len(SCENARIO_TEMPLATES)]

            result = generate_single_example(scenario)

            if result:
                all_examples.append(flatten_example(result))
                failed_count = 0  # Reset fail counter on success
                pbar.update(1)

                # Save checkpoint every 100 examples
                if len(all_examples) % 100 == 0:
                    _save_checkpoint(all_examples, len(all_examples))

                # Rate limiting — Gemini free tier: 15 requests/min
                time.sleep(4.5)
            else:
                failed_count += 1

    print(f"\n✅ Gemini generated: {len(all_examples)} examples")

    # ── Phase 2: Rule-Based Padding ───────────────────────────────────────
    remaining = target_count - len(all_examples)
    if remaining > 0:
        print(f"\n📋 Phase 2: Generating {remaining} rule-based examples...")
        rule_examples = generate_rule_based_examples(remaining)
        all_examples.extend(rule_examples)
        print(f"✅ Rule-based generated: {len(rule_examples)} examples")

    # ── Save Final Dataset ────────────────────────────────────────────────
    print(f"\n💾 Saving dataset...")

    df = pd.DataFrame(all_examples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Full dataset
    full_path = "data/dataset.csv"
    df.to_csv(full_path, index=False)

    # Sample (first 50 rows for quick review)
    sample_path = "data/dataset_sample.csv"
    df.head(50).to_csv(sample_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Dataset Generation Complete!")
    print("=" * 60)
    print(f"  Total rows     : {len(df)}")
    print(f"  Full dataset   : {full_path}")
    print(f"  Sample file    : {sample_path}")
    print(f"\n  Scenario breakdown:")
    print(df["scenario_type"].value_counts().to_string())
    print("=" * 60)

    return df


def _save_checkpoint(examples: list, count: int):
    """Save intermediate checkpoint during generation."""
    os.makedirs("data", exist_ok=True)
    checkpoint_path = f"data/checkpoint_{count}.csv"
    pd.DataFrame(examples).to_csv(checkpoint_path, index=False)
    print(f"\n  💾 Checkpoint saved: {checkpoint_path}")


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(target_count=1000)

    # Print 3 sample rows for visual confirmation
    print("\n📋 Sample rows from dataset:\n")
    for i, row in df.head(3).iterrows():
        print(f"--- Row {i+1} ---")
        print(f"Scenario  : {row['scenario_type']}")
        print(f"Input     : {row['input_text'][:120]}...")
        print(f"Response  : {row['target_response'][:120]}...")
        print()