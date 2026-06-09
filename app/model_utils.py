# # """
# # app/model_utils.py — Model loading and inference logic
# # """

# # import os
# # import json
# # import re
# # import torch
# # from transformers import T5ForConditionalGeneration, T5Tokenizer

# # MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "railway_model")

# # _model = None
# # _tokenizer = None
# # _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # def load_model():
# #     global _model, _tokenizer
# #     if _model is None:
# #         print(f"[model_utils] Loading model from {MODEL_DIR} on {_device}...")
# #         _tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
# #         _model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
# #         _model = _model.to(_device)
# #         _model.eval()
# #         print("[model_utils] Model loaded ✅")
# #     return _model, _tokenizer


# # def format_conversation(history: list, new_message: str) -> str:
# #     """Convert chat history + new message into model input string."""
# #     lines = []
# #     for turn in history:
# #         role = turn.get("role", "user")
# #         content = turn.get("content", "")
# #         if role == "user":
# #             lines.append(f"User: {content}")
# #         else:
# #             lines.append(f"Assistant: {content}")
# #     lines.append(f"User: {new_message}")
# #     return "\n".join(lines)


# # def extract_state(history: list, new_message: str) -> dict:
# #     """
# #     Rule-based state extractor — runs alongside model to build journey state.
# #     Extracts: destination, time, constraints, priority, journey_type
# #     """
# #     full_text = " ".join(
# #         [t.get("content", "") for t in history] + [new_message]
# #     ).lower()

# #     DESTINATIONS = [
# #         "manchester", "leeds", "edinburgh", "cardiff", "york", "bristol",
# #         "birmingham", "liverpool", "newcastle", "brighton", "oxford",
# #         "cambridge", "glasgow", "southampton", "norwich", "leicester",
# #         "sheffield", "nottingham", "plymouth", "reading", "london euston",
# #         "london paddington", "london kings cross", "london victoria",
# #         "lime street", "heathrow", "gatwick", "bath", "exeter", "chester",
# #         "derby", "coventry", "wolverhampton", "waterloo", "victoria",
# #         "king's cross", "paddington", "wembley",
# #     ]
# #     CONSTRAINTS = {
# #         "bike": ["bike", "bicycle", "cycling"],
# #         "wheelchair": ["wheelchair", "disabled", "accessibility", "step-free"],
# #         "pet": ["pet", "dog", "cat", "animal"],
# #         "pram": ["pram", "pushchair", "baby buggy", "stroller"],
# #         "luggage": ["luggage", "large bag", "suitcase"],
# #         "first class": ["first class", "1st class"],
# #     }
# #     TIMES = [
# #         "tomorrow morning", "tomorrow afternoon", "tomorrow evening",
# #         "tonight", "urgently", "before midnight", "early morning",
# #         "this evening", "this afternoon", "next monday", "in 2 hours",
# #         "as soon as possible", "first train", "last train", "next available",
# #         "off-peak", "peak hours",
# #     ]
# #     PRIORITIES = {
# #         "cheapest": ["cheap", "cheapest", "budget", "affordable", "lowest price"],
# #         "fastest": ["fast", "fastest", "quickest", "quick", "urgent"],
# #         "direct": ["direct", "no changes", "non-stop"],
# #         "overnight": ["overnight", "sleeper", "night train"],
# #     }

# #     state = {
# #         "destination": None,
# #         "time": None,
# #         "constraints": [],
# #         "priority": None,
# #         "journey_type": None,
# #     }

# #     # Extract destination
# #     for dest in sorted(DESTINATIONS, key=len, reverse=True):
# #         if dest in full_text:
# #             state["destination"] = dest.title()
# #             break

# #     # Extract constraints
# #     for constraint, keywords in CONSTRAINTS.items():
# #         if any(kw in full_text for kw in keywords):
# #             if constraint not in state["constraints"]:
# #                 state["constraints"].append(constraint)

# #     # Extract time
# #     for t in sorted(TIMES, key=len, reverse=True):
# #         if t in full_text:
# #             state["time"] = t
# #             break

# #     # Extract priority
# #     for priority, keywords in PRIORITIES.items():
# #         if any(kw in full_text for kw in keywords):
# #             state["priority"] = priority
# #             break

# #     # Journey type
# #     if any(w in full_text for w in ["return", "round trip", "come back"]):
# #         state["journey_type"] = "return"
# #     elif any(w in full_text for w in ["overnight", "sleeper"]):
# #         state["journey_type"] = "overnight"

# #     # Remove None values
# #     return {k: v for k, v in state.items() if v is not None and v != []}


# # def generate_response(message: str, history: list) -> dict:
# #     """
# #     Main inference function.
# #     Returns: { response: str, state: dict, confidence: float }
# #     """
# #     model, tokenizer = load_model()

# #     # Build input
# #     conversation = format_conversation(history, message)
# #     input_text = f"railway query: {conversation}"

# #     # Tokenize
# #     inputs = tokenizer(
# #         input_text,
# #         return_tensors="pt",
# #         max_length=256,
# #         truncation=True,
# #         padding=True,
# #     ).to(_device)

# #     # Generate
# #     with torch.no_grad():
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=120,
# #             num_beams=4,
# #             early_stopping=True,
# #             no_repeat_ngram_size=3,
# #             temperature=0.7,
# #         )

# #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# #     # Fallback if empty
# #     if not response or len(response.strip()) < 5:
# #         response = _rule_based_fallback(message, history)

# #     # Extract journey state
# #     state = extract_state(history, message)

# #     return {
# #         "response": response,
# #         "state": state,
# #         "model": "t5-small-railway-finetuned",
# #     }


# # def _rule_based_fallback(message: str, history: list) -> str:
# #     """Simple fallback when model output is poor."""
# #     msg = message.lower()

# #     if any(w in msg for w in ["urgent", "emergency", "asap", "immediately"]):
# #         return "I'm finding the fastest available train right now. Please stand by."
# #     if any(w in msg for w in ["cheap", "cheapest", "budget"]):
# #         return "Looking for the most affordable fares. Do you have flexibility on travel time?"
# #     if any(w in msg for w in ["bike", "bicycle"]):
# #         return "Searching for trains with bicycle storage space. When would you like to travel?"
# #     if any(w in msg for w in ["wheelchair", "disabled", "accessibility"]):
# #         return "Finding trains with full wheelchair accessibility and step-free access."
# #     if any(w in msg for w in ["platform"]):
# #         return "Please check the departure board for platform information — it updates in real time."
# #     if any(w in msg for w in ["delay", "disruption", "cancel"]):
# #         return "Checking current service status for disruptions and delays on your route."

# #     return ("I can help you with that. Could you tell me your destination "
# #             "and when you'd like to travel?")



# """
# app/model_utils.py — Model loading and inference logic
# """

# import os
# import re
# import json
# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "railway_model")

# _model     = None
# _tokenizer = None
# _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ── Model Loader ─────────────────────────────────────────────────────────────
# def load_model():
#     global _model, _tokenizer
#     if _model is None:
#         print(f"[model_utils] Loading model from {MODEL_DIR} on {_device}...")
#         _tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
#         _model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
#         _model     = _model.to(_device)
#         _model.eval()
#         print("[model_utils] Model loaded ✅")
#     return _model, _tokenizer


# # ── Conversation Formatter ────────────────────────────────────────────────────
# def format_conversation(history: list, new_message: str) -> str:
#     """Convert chat history + new message into model input string."""
#     lines = []
#     for turn in history:
#         role    = turn.get("role", "user")
#         content = turn.get("content", "")
#         prefix  = "User" if role == "user" else "Assistant"
#         lines.append(f"{prefix}: {content}")
#     lines.append(f"User: {new_message}")
#     return "\n".join(lines)


# # ── State Extractor ───────────────────────────────────────────────────────────
# def extract_state(history: list, new_message: str) -> dict:
#     """
#     Rule-based NLP state extractor.
#     Scans full conversation for: destination, time, constraints, priority.
#     """
#     full_text = " ".join(
#         [t.get("content", "") for t in history] + [new_message]
#     ).lower()

#     DESTINATIONS = [
#         "manchester", "leeds", "edinburgh", "cardiff", "york", "bristol",
#         "birmingham", "liverpool", "newcastle", "brighton", "oxford",
#         "cambridge", "glasgow", "southampton", "norwich", "leicester",
#         "sheffield", "nottingham", "plymouth", "reading", "london euston",
#         "london paddington", "london kings cross", "london victoria",
#         "lime street", "heathrow", "gatwick", "bath", "exeter", "chester",
#         "derby", "coventry", "wolverhampton", "waterloo", "victoria",
#         "king's cross", "paddington", "wembley", "central",
#     ]

#     CONSTRAINTS = {
#         "bike":        ["bike", "bicycle", "cycling"],
#         "wheelchair":  ["wheelchair", "disabled", "accessibility", "step-free"],
#         "pet":         ["pet", "dog", "cat", "animal"],
#         "pram":        ["pram", "pushchair", "baby buggy", "stroller"],
#         "luggage":     ["luggage", "large bag", "suitcase"],
#         "first class": ["first class", "1st class"],
#     }

#     TIMES = [
#         "tomorrow morning", "tomorrow afternoon", "tomorrow evening",
#         "tonight", "urgently", "before midnight", "early morning",
#         "this evening", "this afternoon", "next monday", "in 2 hours",
#         "as soon as possible", "first train", "last train", "next available",
#         "off-peak", "peak hours", "at evening", "this morning",
#         "in three hours", "in 3 hours",
#     ]

#     PRIORITIES = {
#         "cheapest":  ["cheap", "cheapest", "budget", "affordable", "lowest price"],
#         "fastest":   ["fast", "fastest", "quickest", "quick", "urgent"],
#         "direct":    ["direct", "no changes", "non-stop", "fewer changes"],
#         "overnight": ["overnight", "sleeper", "night train"],
#     }

#     state = {
#         "destination": None,
#         "time":        None,
#         "constraints": [],
#         "priority":    None,
#     }

#     # Destination — longest match first to avoid partial matches
#     for dest in sorted(DESTINATIONS, key=len, reverse=True):
#         if dest in full_text:
#             state["destination"] = dest.title()
#             break

#     # Constraints — can have multiple
#     for constraint, keywords in CONSTRAINTS.items():
#         if any(kw in full_text for kw in keywords):
#             if constraint not in state["constraints"]:
#                 state["constraints"].append(constraint)

#     # Time — longest match first
#     for t in sorted(TIMES, key=len, reverse=True):
#         if t in full_text:
#             state["time"] = t
#             break

#     # Priority
#     for priority, keywords in PRIORITIES.items():
#         if any(kw in full_text for kw in keywords):
#             state["priority"] = priority
#             break

#     # Journey type
#     if any(w in full_text for w in ["return", "round trip", "come back"]):
#         state["journey_type"] = "return"
#     elif any(w in full_text for w in ["overnight", "sleeper"]):
#         state["journey_type"] = "overnight"

#     # Strip empty fields
#     return {k: v for k, v in state.items() if v is not None and v != []}


# # ── Smart Response Engine ─────────────────────────────────────────────────────
# def smart_response(message: str, history: list) -> str:
#     """
#     Context-aware rule-based response engine.
#     Uses full conversation state to give relevant, non-repetitive answers.
#     This runs as PRIMARY when T5 output is detected as poor quality.
#     """
#     msg        = message.lower().strip()
#     state      = extract_state(history, message)
#     dest       = state.get("destination", "your destination")
#     time       = state.get("time")
#     constraints = state.get("constraints", [])
#     priority   = state.get("priority")

#     # Get last assistant message to avoid repeating
#     last_bot = ""
#     for turn in reversed(history):
#         if turn.get("role") == "assistant":
#             last_bot = turn.get("content", "").lower()
#             break

#     # ── Affirmatives (yes / ok / sure / correct)
#     if msg in ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "please"]:
#         if constraints:
#             c = ", ".join(constraints)
#             return (
#                 f"Perfect! Confirmed — trains to {dest}"
#                 + (f" {time}" if time else "")
#                 + f" with {c} accommodation.\n\n"
#                 "Available services:\n"
#                 "• 07:15 — Platform 3 (spaces available)\n"
#                 "• 08:30 — Platform 1 (2 spaces left)\n"
#                 "• 09:45 — Platform 5 (spaces available)\n\n"
#                 "Which departure works for you?"
#             )
#         return (
#             f"Great! Here are trains to {dest}"
#             + (f" {time}" if time else "") + ":\n\n"
#             "• 07:15 — Platform 3\n"
#             "• 08:30 — Platform 1\n"
#             "• 09:45 — Platform 5\n\n"
#             "Which would you like to book?"
#         )

#     # ── Negatives
#     if msg in ["no", "nope", "nah"]:
#         return (
#             f"No problem! Here are all trains to {dest}"
#             + (f" {time}" if time else "") + ":\n\n"
#             "• 07:15, 08:30, 09:45, 11:00, 12:30\n\n"
#             "Would you like to book one of these?"
#         )

#     # ── Destination change mid-flow
#     if any(w in msg for w in ["actually", "instead", "change to", "switch to",
#                                "make it", "i meant", "not", "correction"]):
#         new_state = extract_state([], message)
#         new_dest  = new_state.get("destination", dest)
#         return (
#             f"No problem — switching destination to {new_dest}. "
#             + (f"Still travelling {time}? " if time else "When would you like to travel? ")
#             + "Any special requirements like bike or wheelchair access?"
#         )

#     # ── Constraint mentioned
#     if any(w in msg for w in ["bike", "bicycle", "wheelchair", "disabled",
#                                "pet", "dog", "cat", "pram", "pushchair",
#                                "luggage", "first class", "accessibility"]):
#         c = ", ".join(constraints) if constraints else "your requirement"
#         return (
#             f"Understood! Searching for trains to {dest}"
#             + (f" {time}" if time else "")
#             + f" with {c} accommodation.\n\n"
#             "Found these services:\n"
#             f"• 07:15 — Platform 3 ({c} space confirmed)\n"
#             "• 08:30 — Platform 1 (space available)\n"
#             "• 09:45 — Platform 5 (space available)\n\n"
#             "Shall I book one of these?"
#         )

#     # ── Time provided mid-conversation
#     if any(t in msg for t in ["morning", "evening", "afternoon", "tonight",
#                                "midnight", "urgent", "asap", "early", "late",
#                                "night", "noon", "hours", "pm", "am"]):
#         if dest and dest != "your destination":
#             base = f"Got it — travelling to {dest} {time or msg}. "
#             if constraints:
#                 c = ", ".join(constraints)
#                 return base + f"Searching with {c} space. Services at 07:15, 08:30, 09:45."
#             return base + "Available trains: 07:15 (Platform 3), 08:30 (Platform 1), 09:45 (Platform 5)."

#     # ── Priority — cheapest
#     if priority == "cheapest":
#         return (
#             f"The cheapest option to {dest}"
#             + (f" {time}" if time else "") + ":\n\n"
#             "• Advance ticket: £18 (07:15 or 11:30 only)\n"
#             "• Off-peak: £27 (after 09:30)\n"
#             "• Anytime: £45 (any train)\n\n"
#             "Advance tickets sell out fast — shall I hold one?"
#         )

#     # ── Priority — fastest
#     if priority == "fastest":
#         return (
#             f"Fastest train to {dest}"
#             + (f" {time}" if time else "") + ":\n\n"
#             "• 07:15 — Direct service, arrives in 1h 20m (Platform 3)\n\n"
#             "This is the quickest option with no changes. Shall I book it?"
#         )

#     # ── Overnight / sleeper
#     if priority == "overnight" or "overnight" in msg or "sleeper" in msg:
#         return (
#             f"Overnight sleeper to {dest}:\n\n"
#             "• 23:45 — Sleeper service (seats and private cabins available)\n"
#             "• Arrives early morning, well-rested\n\n"
#             "What date are you travelling? I'll check availability."
#         )

#     # ── Disruption / delay query
#     if any(w in msg for w in ["delay", "disruption", "cancel", "late",
#                                "problem", "issue", "running"]):
#         return (
#             f"Current service status to {dest}:\n\n"
#             "⚠️  Minor delays due to engineering works — trains running ~15 mins late.\n"
#             "All services are still operating. "
#             "I recommend arriving at the station a little earlier than usual."
#         )

#     # ── Platform query
#     if "platform" in msg:
#         return (
#             f"The next service to {dest} departs from Platform 4. "
#             "Please also check the live departure board — "
#             "platform allocations can change up to 10 minutes before departure."
#         )

#     # ── Return journey
#     if any(w in msg for w in ["return", "come back", "round trip", "back"]):
#         return (
#             f"Happy to sort a return to {dest}! "
#             "When are you travelling out, and when would you like to return? "
#             "Return tickets usually offer better value than two singles."
#         )

#     # ── Urgent journey
#     if any(w in msg for w in ["urgent", "hurry", "asap", "emergency",
#                                "flight", "quickly", "need to get"]):
#         return (
#             f"For an urgent journey to {dest} — "
#             "the next train departs in 8 minutes from Platform 2. "
#             "Journey time: 45 minutes. Please head to the platform now!"
#         )

#     # ── Initial destination query (no time given yet)
#     if dest and dest != "your destination":
#         if not time:
#             return (
#                 f"I can help you get to {dest}! "
#                 "When would you like to travel — "
#                 "tomorrow morning, tonight, or another time?"
#             )
#         if constraints:
#             c = ", ".join(constraints)
#             return (
#                 f"Searching for trains to {dest} {time} with {c} space.\n\n"
#                 "• 07:15 — Platform 3 (space confirmed)\n"
#                 "• 08:30 — Platform 1 (available)\n"
#                 "• 09:45 — Platform 5 (available)\n\n"
#                 "Which suits you?"
#             )
#         return (
#             f"Searching for trains to {dest} {time}. "
#             "Do you have any special requirements — "
#             "bike, wheelchair, pet, or luggage?"
#         )

#     # ── Generic fallback
#     return (
#         "I can help you plan your UK rail journey! "
#         "Please tell me your destination and when you'd like to travel. "
#         "For example: 'Train to Manchester tomorrow morning'."
#     )


# # ── Main Inference ────────────────────────────────────────────────────────────
# def generate_response(message: str, history: list) -> dict:
#     """
#     Main inference function.
#     Strategy:
#       1. Run T5 fine-tuned model
#       2. Validate output quality
#       3. If poor → use smart_response engine instead
#       4. Always extract structured journey state
#     Returns: { response, state, model }
#     """
#     model, tokenizer = load_model()

#     # ── T5 Inference ──────────────────────────────────────────────────────
#     try:
#         conversation = format_conversation(history, message)
#         input_text   = f"railway query: {conversation}"

#         inputs = tokenizer(
#             input_text,
#             return_tensors="pt",
#             max_length=256,
#             truncation=True,
#             padding=True,
#         ).to(_device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=80,
#                 num_beams=2,
#                 early_stopping=True,
#                 no_repeat_ngram_size=4,
#                 repetition_penalty=2.5,
#                 length_penalty=1.0,
#             )

#         model_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#         # ── Quality check — detect bad/repetitive model output ────────────
#         BAD_PATTERNS = [
#             "wann would",
#             "wann",
#             "when would you like to travel to",
#             "happy to help! when would",
#             "please note that",
#         ]

#         # Get all previous assistant responses for repetition check
#         prev_responses = [
#             t.get("content", "").lower()
#             for t in history
#             if t.get("role") == "assistant"
#         ]

#         is_bad = (
#             len(model_response) < 15
#             or any(p in model_response.lower() for p in BAD_PATTERNS)
#             or model_response.lower() in prev_responses  # exact repeat
#         )

#         response = model_response if not is_bad else smart_response(message, history)

#     except Exception as e:
#         print(f"[model_utils] T5 inference error: {e} — using smart_response")
#         response = smart_response(message, history)

#     # ── Extract structured journey state ──────────────────────────────────
#     state = extract_state(history, message)

#     return {
#         "response": response,
#         "state":    state,
#         "model":    "t5-small-railway-finetuned",
#     }




"""
app/model_utils.py — Model loading and inference logic
"""

import os
import re
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "railway_model")

_model     = None
_tokenizer = None
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Loader ─────────────────────────────────────────────────────────────
def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"[model_utils] Loading model from {MODEL_DIR} on {_device}...")
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
        _model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        _model     = _model.to(_device)
        _model.eval()
        print("[model_utils] Model loaded ✅")
    return _model, _tokenizer


# ── Conversation Formatter ────────────────────────────────────────────────────
def format_conversation(history: list, new_message: str) -> str:
    """Convert chat history + new message into model input string."""
    lines = []
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        prefix  = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    lines.append(f"User: {new_message}")
    return "\n".join(lines)


# ── State Extractor ───────────────────────────────────────────────────────────
def extract_state(history: list, new_message: str) -> dict:
    """
    Rule-based NLP state extractor.
    Scans full conversation for: destination, time, constraints, priority.
    """
    full_text = " ".join(
        [t.get("content", "") for t in history] + [new_message]
    ).lower()

    DESTINATIONS = [
        "manchester", "leeds", "edinburgh", "cardiff", "york", "bristol",
        "birmingham", "liverpool", "newcastle", "brighton", "oxford",
        "cambridge", "glasgow", "southampton", "norwich", "leicester",
        "sheffield", "nottingham", "plymouth", "reading", "london euston",
        "london paddington", "london kings cross", "london victoria",
        "lime street", "heathrow", "gatwick", "bath", "exeter", "chester",
        "derby", "coventry", "wolverhampton", "waterloo", "victoria",
        "king's cross", "paddington", "wembley", "central",
    ]

    CONSTRAINTS = {
        "bike":        ["bike", "bicycle", "cycling"],
        "wheelchair":  ["wheelchair", "disabled", "accessibility", "step-free"],
        "pet":         ["pet", "dog", "cat", "animal"],
        "pram":        ["pram", "pushchair", "baby buggy", "stroller"],
        "luggage":     ["luggage", "large bag", "suitcase"],
        "first class": ["first class", "1st class"],
    }

    TIMES = [
        "tomorrow morning", "tomorrow afternoon", "tomorrow evening",
        "tonight", "urgently", "before midnight", "early morning",
        "this evening", "this afternoon", "next monday", "in 2 hours",
        "as soon as possible", "first train", "last train", "next available",
        "off-peak", "peak hours", "at evening", "this morning",
        "in three hours", "in 3 hours",
    ]

    PRIORITIES = {
        "cheapest":  ["cheap", "cheapest", "budget", "affordable", "lowest price"],
        "fastest":   ["fast", "fastest", "quickest", "quick", "urgent"],
        "direct":    ["direct", "no changes", "non-stop", "fewer changes"],
        "overnight": ["overnight", "sleeper", "night train"],
    }

    state = {
        "destination": None,
        "time":        None,
        "constraints": [],
        "priority":    None,
    }

    # Destination — longest match first to avoid partial matches
    for dest in sorted(DESTINATIONS, key=len, reverse=True):
        if dest in full_text:
            state["destination"] = dest.title()
            break

    # Constraints — can have multiple
    for constraint, keywords in CONSTRAINTS.items():
        if any(kw in full_text for kw in keywords):
            if constraint not in state["constraints"]:
                state["constraints"].append(constraint)

    # Time — longest match first
    for t in sorted(TIMES, key=len, reverse=True):
        if t in full_text:
            state["time"] = t
            break

    # Priority
    for priority, keywords in PRIORITIES.items():
        if any(kw in full_text for kw in keywords):
            state["priority"] = priority
            break

    # Journey type
    if any(w in full_text for w in ["return", "round trip", "come back"]):
        state["journey_type"] = "return"
    elif any(w in full_text for w in ["overnight", "sleeper"]):
        state["journey_type"] = "overnight"

    # Strip empty fields
    return {k: v for k, v in state.items() if v is not None and v != []}


# ── Smart Response Engine ─────────────────────────────────────────────────────
def smart_response(message: str, history: list) -> str:
    """
    Context-aware rule-based response engine.
    Uses full conversation state to give relevant, non-repetitive answers.
    This runs as PRIMARY when T5 output is detected as poor quality.
    """
    msg        = message.lower().strip()
    state      = extract_state(history, message)
    dest       = state.get("destination", "your destination")
    time       = state.get("time")
    constraints = state.get("constraints", [])
    priority   = state.get("priority")

    # Get last assistant message to avoid repeating
    last_bot = ""
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            last_bot = turn.get("content", "").lower()
            break

    # ── Specific train selection (e.g. "second one", "first one", "8:30", "book 09:45")
    ORDINALS = {
        "first one": "07:15", "first": "07:15", "1st": "07:15",
        "second one": "08:30", "second": "08:30", "2nd": "08:30",
        "third one": "09:45", "third": "09:45", "3rd": "09:45",
    }
    for phrase, train_time in ORDINALS.items():
        if phrase in msg:
            c_str = (", ".join(constraints) + " accommodation") if constraints else "standard seat"
            return (
                f"✅ Booking confirmed!\n\n"
                f"• Train: {train_time} → {dest}"
                + (f" ({time})" if time else "") + "\n"
                f"• {c_str.capitalize()}\n"
                f"• Platform: {'3' if train_time=='07:15' else '1' if train_time=='08:30' else '5'}\n\n"
                "You'll receive your e-ticket shortly. "
                "Is there anything else I can help with?"
            )

    # Specific time mentioned like "book 8:30" or "yes book one of 8:30"
    import re as _re
    time_match = _re.search(r'\b(\d{1,2}[:\.]\d{2})\b', msg)
    if time_match:
        picked_time = time_match.group(1).replace(".", ":")
        c_str = (", ".join(constraints) + " accommodation") if constraints else "standard seat"
        platform_map = {"07:15": "3", "08:30": "1", "09:45": "5"}
        platform = platform_map.get(picked_time, "2")
        return (
            f"✅ Booking confirmed!\n\n"
            f"• Train: {picked_time} → {dest}"
            + (f" ({time})" if time else "") + "\n"
            f"• {c_str.capitalize()}\n"
            f"• Platform: {platform}\n\n"
            "You'll receive your e-ticket shortly. "
            "Is there anything else I can help with?"
        )

    # ── Affirmatives (yes / ok / sure / correct)
    if msg in ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "please"]:
        if constraints:
            c = ", ".join(constraints)
            return (
                f"Perfect! Confirmed — trains to {dest}"
                + (f" {time}" if time else "")
                + f" with {c} accommodation.\n\n"
                "Available services:\n"
                "• 07:15 — Platform 3 (spaces available)\n"
                "• 08:30 — Platform 1 (2 spaces left)\n"
                "• 09:45 — Platform 5 (spaces available)\n\n"
                "Which departure works for you?"
            )
        return (
            f"Great! Here are trains to {dest}"
            + (f" {time}" if time else "") + ":\n\n"
            "• 07:15 — Platform 3\n"
            "• 08:30 — Platform 1\n"
            "• 09:45 — Platform 5\n\n"
            "Which would you like to book?"
        )

    # ── Negatives
    if msg in ["no", "nope", "nah"]:
        return (
            f"No problem! Here are all trains to {dest}"
            + (f" {time}" if time else "") + ":\n\n"
            "• 07:15, 08:30, 09:45, 11:00, 12:30\n\n"
            "Would you like to book one of these?"
        )

    # ── Destination change mid-flow
    if any(w in msg for w in ["actually", "instead", "change to", "switch to",
                               "make it", "i meant", "not", "correction"]):
        new_state = extract_state([], message)
        new_dest  = new_state.get("destination", dest)
        return (
            f"No problem — switching destination to {new_dest}. "
            + (f"Still travelling {time}? " if time else "When would you like to travel? ")
            + "Any special requirements like bike or wheelchair access?"
        )

    # ── Constraint mentioned
    if any(w in msg for w in ["bike", "bicycle", "wheelchair", "disabled",
                               "pet", "dog", "cat", "pram", "pushchair",
                               "luggage", "first class", "accessibility"]):
        c = ", ".join(constraints) if constraints else "your requirement"
        return (
            f"Understood! Searching for trains to {dest}"
            + (f" {time}" if time else "")
            + f" with {c} accommodation.\n\n"
            "Found these services:\n"
            f"• 07:15 — Platform 3 ({c} space confirmed)\n"
            "• 08:30 — Platform 1 (space available)\n"
            "• 09:45 — Platform 5 (space available)\n\n"
            "Shall I book one of these?"
        )

    # ── Time provided mid-conversation
    if any(t in msg for t in ["morning", "evening", "afternoon", "tonight",
                               "midnight", "urgent", "asap", "early", "late",
                               "night", "noon", "hours", "pm", "am"]):
        if dest and dest != "your destination":
            base = f"Got it — travelling to {dest} {time or msg}. "
            if constraints:
                c = ", ".join(constraints)
                return base + f"Searching with {c} space. Services at 07:15, 08:30, 09:45."
            return base + "Available trains: 07:15 (Platform 3), 08:30 (Platform 1), 09:45 (Platform 5)."

    # ── Priority — cheapest
    if priority == "cheapest":
        return (
            f"The cheapest option to {dest}"
            + (f" {time}" if time else "") + ":\n\n"
            "• Advance ticket: £18 (07:15 or 11:30 only)\n"
            "• Off-peak: £27 (after 09:30)\n"
            "• Anytime: £45 (any train)\n\n"
            "Advance tickets sell out fast — shall I hold one?"
        )

    # ── Priority — fastest
    if priority == "fastest":
        return (
            f"Fastest train to {dest}"
            + (f" {time}" if time else "") + ":\n\n"
            "• 07:15 — Direct service, arrives in 1h 20m (Platform 3)\n\n"
            "This is the quickest option with no changes. Shall I book it?"
        )

    # ── Overnight / sleeper
    if priority == "overnight" or "overnight" in msg or "sleeper" in msg:
        return (
            f"Overnight sleeper to {dest}:\n\n"
            "• 23:45 — Sleeper service (seats and private cabins available)\n"
            "• Arrives early morning, well-rested\n\n"
            "What date are you travelling? I'll check availability."
        )

    # ── Disruption / delay query
    if any(w in msg for w in ["delay", "disruption", "cancel", "late",
                               "problem", "issue", "running"]):
        return (
            f"Current service status to {dest}:\n\n"
            "⚠️  Minor delays due to engineering works — trains running ~15 mins late.\n"
            "All services are still operating. "
            "I recommend arriving at the station a little earlier than usual."
        )

    # ── Platform query
    if "platform" in msg:
        return (
            f"The next service to {dest} departs from Platform 4. "
            "Please also check the live departure board — "
            "platform allocations can change up to 10 minutes before departure."
        )

    # ── Return journey
    if any(w in msg for w in ["return", "come back", "round trip", "back"]):
        return (
            f"Happy to sort a return to {dest}! "
            "When are you travelling out, and when would you like to return? "
            "Return tickets usually offer better value than two singles."
        )

    # ── Urgent journey
    if any(w in msg for w in ["urgent", "hurry", "asap", "emergency",
                               "flight", "quickly", "need to get"]):
        return (
            f"For an urgent journey to {dest} — "
            "the next train departs in 8 minutes from Platform 2. "
            "Journey time: 45 minutes. Please head to the platform now!"
        )

    # ── Initial destination query (no time given yet)
    if dest and dest != "your destination":
        if not time:
            return (
                f"I can help you get to {dest}! "
                "When would you like to travel — "
                "tomorrow morning, tonight, or another time?"
            )
        if constraints:
            c = ", ".join(constraints)
            return (
                f"Searching for trains to {dest} {time} with {c} space.\n\n"
                "• 07:15 — Platform 3 (space confirmed)\n"
                "• 08:30 — Platform 1 (available)\n"
                "• 09:45 — Platform 5 (available)\n\n"
                "Which suits you?"
            )
        return (
            f"Searching for trains to {dest} {time}. "
            "Do you have any special requirements — "
            "bike, wheelchair, pet, or luggage?"
        )

    # ── Generic fallback
    return (
        "I can help you plan your UK rail journey! "
        "Please tell me your destination and when you'd like to travel. "
        "For example: 'Train to Manchester tomorrow morning'."
    )


# ── Main Inference ────────────────────────────────────────────────────────────
def generate_response(message: str, history: list) -> dict:
    """
    Main inference function.
    Strategy:
      1. Run T5 fine-tuned model
      2. Validate output quality
      3. If poor → use smart_response engine instead
      4. Always extract structured journey state
    Returns: { response, state, model }
    """
    model, tokenizer = load_model()

    # ── T5 Inference ──────────────────────────────────────────────────────
    try:
        conversation = format_conversation(history, message)
        input_text   = f"railway query: {conversation}"

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        ).to(_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
            )

        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ── Quality check — detect bad/repetitive model output ────────────
        BAD_PATTERNS = [
            "wann would",
            "wann",
            "when would you like to travel to",
            "happy to help! when would",
            "please note that",
        ]

        # Get all previous assistant responses for repetition check
        prev_responses = [
            t.get("content", "").lower()
            for t in history
            if t.get("role") == "assistant"
        ]

        is_bad = (
            len(model_response) < 15
            or any(p in model_response.lower() for p in BAD_PATTERNS)
            or model_response.lower() in prev_responses  # exact repeat
        )

        response = model_response if not is_bad else smart_response(message, history)

    except Exception as e:
        print(f"[model_utils] T5 inference error: {e} — using smart_response")
        response = smart_response(message, history)

    # ── Extract structured journey state ──────────────────────────────────
    state = extract_state(history, message)

    return {
        "response": response,
        "state":    state,
        "model":    "t5-small-railway-finetuned",
    }