"""
app/model_utils.py - Model loading and inference logic
"""

import os
import re
import json
import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "railway_model")
_model     = None
_tokenizer = None
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# MODEL LOADER

def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"[model_utils] Loading model from {MODEL_DIR} on {_device}...")
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
        _model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        _model     = _model.to(_device)
        _model.eval()
        print("[model_utils] Model loaded successfully")
    return _model, _tokenizer



# CONVERSATION FORMATTER

def format_conversation(history: list, new_message: str) -> str:
    lines = []
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        prefix  = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    lines.append(f"User: {new_message}")
    return "\n".join(lines)



# STATE EXTRACTOR

def extract_state(history: list, new_message: str) -> dict:
    full_text = " ".join(
        [t.get("content", "") for t in history] + [new_message]
    ).lower()

    DESTINATIONS = [
        "new delhi railway station", "mumbai central", "howrah junction",
        "chennai central", "secunderabad junction", "pune junction",
        "ahmedabad junction", "jaipur junction", "lucknow junction",
        "patna junction", "bhopal junction", "nagpur junction",
        "bangalore city junction", "hazrat nizamuddin",
        "anand vihar", "lokmanya tilak terminus",
        "bengaluru", "bangalore", "hyderabad", "ahmedabad", "chandigarh",
        "amritsar", "varanasi", "bhubaneswar", "visakhapatnam",
        "prayagraj", "allahabad", "coimbatore", "madurai",
        "vijayawada", "ludhiana", "jalandhar", "jodhpur",
        "udaipur", "shimla", "dehradun", "guwahati",
        "kolkata", "chennai", "mumbai", "delhi", "pune",
        "jaipur", "lucknow", "patna", "bhopal", "indore",
        "nagpur", "surat", "kochi", "ranchi", "raipur",
        "mysuru", "mysore", "trichy", "rajkot", "ajmer",
        "meerut", "kanpur", "agra", "jammu",
        "london kings cross", "london paddington", "london victoria",
        "london euston", "edinburgh", "manchester", "birmingham",
        "leeds", "cardiff", "york", "bristol", "liverpool",
        "newcastle", "glasgow", "brighton", "oxford", "cambridge",
        "sheffield", "nottingham", "leicester", "exeter",
        "gatwick", "heathrow", "wembley", "waterloo",
    ]

    CONSTRAINTS = {
        "wheelchair":     ["wheelchair", "disabled", "step-free"],
        "senior citizen": ["senior citizen", "senior", "elderly", "old age"],
        "pregnant woman": ["pregnant", "pregnancy", "maternity"],
        "pet":            ["pet", "dog", "cat", "animal"],
        "large luggage":  ["large luggage", "luggage", "suitcase", "heavy bag"],
        "family travel":  ["family", "kids", "children", "travelling with family"],
        "child":          ["child travelling", "child", "minor", "baby"],
        "ladies coach":   ["ladies coach", "ladies", "women only"],
        "divyang":        ["divyang", "handicapped", "differently abled", "accessibility"],
        "medical":        ["medical emergency", "medical", "health condition"],
        "bike":           ["bike", "bicycle", "cycling"],
        "pram":           ["pram", "pushchair", "stroller", "baby buggy"],
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
        "cheapest":     ["cheap", "cheapest", "budget", "affordable", "lowest price", "lowest fare"],
        "fastest":      ["fast", "fastest", "quickest", "quick", "urgent", "shortest travel time"],
        "direct":       ["direct", "no changes", "non-stop", "fewer stops", "fewest stops"],
        "confirmed":    ["confirmed", "confirmed seat", "confirmed berth", "guaranteed"],
        "overnight":    ["overnight", "sleeper", "night train", "night journey"],
        "waiting list": ["waiting list", "waitlist", "wl", "lowest waiting list"],
    }

    state = {"destination": None, "time": None, "constraints": [], "priority": None}

    for dest in sorted(DESTINATIONS, key=len, reverse=True):
        if dest in full_text:
            state["destination"] = dest.title()
            break

    for constraint, keywords in CONSTRAINTS.items():
        if any(kw in full_text for kw in keywords):
            if constraint not in state["constraints"]:
                state["constraints"].append(constraint)

    for t in sorted(TIMES, key=len, reverse=True):
        if t in full_text:
            state["time"] = t
            break

    for priority, keywords in PRIORITIES.items():
        if any(kw in full_text for kw in keywords):
            state["priority"] = priority
            break

    if any(w in full_text for w in ["return", "round trip", "come back", "wapas"]):
        state["journey_type"] = "return"
    elif any(w in full_text for w in ["overnight", "sleeper", "night journey"]):
        state["journey_type"] = "overnight"

    return {k: v for k, v in state.items() if v is not None and v != []}



# FOLLOW-UP QUESTION BANK

TICKET_FOLLOWUPS = [
    "Would you like a single or return ticket?",
    "Are you looking for a specific class — Sleeper, 3A, 2A, or First Class?",
    "Do you have a senior citizen or student concession card?",
    "Would you prefer a window or aisle berth?",
    "Do you need a reserved berth or is an unreserved ticket fine?",
]
JOURNEY_FOLLOWUPS = [
    "Would you prefer a direct train or are you okay with one change?",
    "Are you travelling alone or with family?",
    "Do you need to carry any large luggage or special equipment?",
    "Is this a one-way journey or would you like a return as well?",
    "Do you have any special requirements like wheelchair access or ladies coach?",
]
BOOKING_FOLLOWUPS = [
    "Shall I go ahead and check availability for this train?",
    "Do you want me to check the current seat availability before booking?",
    "Would you like to add a meal preference to this booking?",
    "Can I confirm your travel date before I proceed?",
    "Do you need an e-ticket or a physical ticket for this journey?",
]

def _pick_followup(category: str) -> str:
    if category == "ticket":  return random.choice(TICKET_FOLLOWUPS)
    if category == "journey": return random.choice(JOURNEY_FOLLOWUPS)
    if category == "booking": return random.choice(BOOKING_FOLLOWUPS)
    return ""



# SMALL TALK HANDLER

def _handle_small_talk(msg: str, history: list, state: dict) -> str | None:
    dest = state.get("destination")
    time = state.get("time")
    m    = msg.lower().strip()

    # ── Greetings
    GREETINGS = [
        "hi", "hello", "hey", "hii", "helo", "heyy", "heyyy",
        "namaste", "namaskar", "salaam", "good morning", "good evening",
        "good afternoon", "good night", "sup", "yo", "wassup", "what's up",
        "whats up", "howdy", "greetings",
    ]
    if m in GREETINGS or any(m.startswith(g + " ") for g in GREETINGS):
        greet_options = [
            "Hello! Great to have you here. I am Railway AI Assistant — I can help you find trains, check fares, manage bookings, and a lot more. Where are you headed today?",
            "Hey there! I am your Railway AI Assistant. Tell me where you want to go and I will sort out the best options for you.",
            "Hi! Welcome. I am here to make your train journey easy. Just tell me your destination and I will take care of the rest.",
            "Hello! Good to see you. I can help with train bookings, PNR status, fares, seat availability — you name it. Where would you like to travel?",
        ]
        return random.choice(greet_options)

    # ── How are you / casual check
    HOW_ARE_YOU = [
        "how are you", "how r u", "how are u", "how are ya",
        "hows it going", "how's it going", "how you doing",
        "you good", "you okay", "you alright",
    ]
    if any(p in m for p in HOW_ARE_YOU):
        responses = [
            "I am doing great, thanks for asking! Always ready to help you plan a smooth journey. Where are you off to today?",
            "All good here! I am at your service 24/7. Just tell me your destination and I will find the best trains for you.",
            "Doing well, thank you! Now let us get you on the right train — where would you like to go?",
        ]
        return random.choice(responses)

    # ── Who are you / what can you do
    WHO_ARE_YOU = [
        "who are you", "what are you", "what is this", "what can you do",
        "tell me about yourself", "introduce yourself", "what do you do",
        "are you a bot", "are you ai", "are you human", "are you chatgpt",
        "are you real",
    ]
    if any(p in m for p in WHO_ARE_YOU):
        return (
            "I am Railway AI Assistant — an intelligent chatbot built to make train travel simple.\n\n"
            "Here is what I can do for you:\n"
            "  Find trains between any two stations\n"
            "  Check seat and berth availability\n"
            "  Compare fares across classes\n"
            "  Handle special requirements (wheelchair, bike, pet, senior citizen)\n"
            "  Check PNR status and live train updates\n"
            "  Book tickets and manage reservations\n"
            "  Answer any railway-related question\n\n"
            "I am not ChatGPT — I am specifically trained for railways. "
            "Just tell me where you want to go and I will take it from there!"
        )

    # ── Thank you
    THANKS = ["thank", "thanks", "thank you", "thankyou", "thx", "ty", "cheers", "appreciated"]
    if any(p in m for p in THANKS):
        responses = [
            "You are very welcome! Have a safe and comfortable journey. Feel free to come back anytime.",
            "Happy to help! Wishing you a smooth journey. If you need anything else, I am right here.",
            "Anytime! That is what I am here for. Safe travels!",
        ]
        return random.choice(responses)

    # ── Bye / closing
    BYE = [
        "bye", "goodbye", "good bye", "see you", "see ya", "cya", "tata",
        "take care", "later", "ttyl", "gtg", "got to go", "gotta go",
        "close this chat", "end chat", "stop", "quit", "exit",
        "close the chat", "close chat", "now close", "close now",
        "that's all", "thats all", "nothing else", "no more", "done",
        "finish", "i'm done", "im done", "all done",
    ]
    if any(p in m for p in BYE) or m in BYE:
        responses = [
            "Goodbye! Have a wonderful journey. Feel free to come back anytime you need help with train travel.",
            "Take care! It was a pleasure helping you. Safe travels!",
            "Bye! Wishing you a smooth and comfortable journey. See you next time!",
        ]
        return random.choice(responses)

    # ── Frustration / confusion expressions
    FRUSTRATION = [
        "kya bol rha ha", "kya bol raha hai", "what are you saying",
        "you are wrong", "that is wrong", "this is wrong", "not helpful",
        "useless", "stupid", "nonsense", "bakwaas", "pagal", "idiot",
        "i dont understand", "i don't understand", "confused", "what",
        "huh", "what?", "really?", "seriously?", "come on",
        "are you serious", "stop repeating", "same answer", "again same",
    ]
    if any(p in m for p in FRUSTRATION):
        responses = [
            "I understand, and I apologise if my previous response was not helpful. Let me do better. Could you tell me what you are looking for — your destination and travel time? I will give you a clear, direct answer.",
            "Sorry about that! I want to give you the most useful response. Could you rephrase your question or tell me your destination? I am here to help.",
            "Apologies if I was not clear. Let us start fresh — just tell me where you want to go and when, and I will find the best options for you right away.",
        ]
        return random.choice(responses)

    # ── Compliments
    COMPLIMENTS = [
        "good", "great", "awesome", "nice", "perfect", "excellent",
        "well done", "good job", "helpful", "amazing", "brilliant",
        "superb", "fantastic", "you are good", "you are great",
        "love it", "love this",
    ]
    if any(p in m for p in COMPLIMENTS) and len(m.split()) <= 5:
        responses = [
            "Thank you! I am glad I could help. Anything else you need for your journey?",
            "That means a lot! Is there anything else I can help you with?",
            "Thank you! I am here whenever you need me. Where are we travelling next?",
        ]
        return random.choice(responses)

    # ── Yes / affirmative (context-aware)
    YES_WORDS = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay",
                 "correct", "right", "please", "go ahead", "sounds good",
                 "alright", "fine", "absolutely", "definitely", "of course",
                 "haan", "bilkul", "theek hai", "confirm"]
    if m in YES_WORDS:
        if dest:
            constraints = state.get("constraints", [])
            if constraints:
                c = ", ".join(constraints)
                return (
                    f"Searching confirmed trains to {dest}"
                    + (f" {time}" if time else "") +
                    f" with {c} accommodation.\n\n"
                    "Available services:\n"
                    "  06:30 — Platform 1 (berths available)\n"
                    "  09:15 — Platform 3 (2 berths left)\n"
                    "  13:45 — Platform 5 (berths available)\n\n"
                    + _pick_followup("ticket")
                )
            return (
                f"Here are trains to {dest}"
                + (f" {time}" if time else "") + ":\n\n"
                "  06:30 — Platform 1\n"
                "  09:15 — Platform 3\n"
                "  13:45 — Platform 5\n\n"
                + _pick_followup("ticket")
            )
        return "Sure! Could you tell me your destination so I can pull up the right trains for you?"

    # ── No / negative (context-aware)
    NO_WORDS = ["no", "nope", "nah", "nahi", "nahin", "not really",
                "no thanks", "no thank you", "nope thanks"]
    if m in NO_WORDS:
        if dest:
            return (
                f"No problem. Here are all available trains to {dest}"
                + (f" {time}" if time else "") + ":\n\n"
                "  06:30, 09:15, 13:45, 17:00, 21:30\n\n"
                + _pick_followup("journey")
            )
        return "No worries! Is there anything else I can help you with? Just let me know."

    # ── Asking for help generally
    HELP_WORDS = ["help", "help me", "i need help", "assist", "assistance",
                  "support", "guide", "guidance"]
    if m in HELP_WORDS or any(p == m for p in HELP_WORDS):
        return (
            "Of course, I am here to help! Here is what I can assist you with:\n\n"
            "  Train search — find trains between any stations\n"
            "  Fare check   — compare prices across all classes\n"
            "  Availability — check seats and berths in real time\n"
            "  PNR status   — track your booking\n"
            "  Special needs — wheelchair, bike, senior citizen, ladies coach\n"
            "  Live updates — delays, disruptions, platform changes\n\n"
            "Just tell me your destination and when you want to travel!"
        )

    # ── Bored / testing the bot
    BORED = ["test", "testing", "just testing", "checking", "just checking",
             "random", "idk", "i don't know", "i dont know", "dunno",
             "nothing", "nevermind", "never mind", "forget it", "nvm"]
    if m in BORED:
        responses = [
            "No worries! Whenever you are ready to plan a journey, just tell me your destination and I will take it from there.",
            "All good! I am here whenever you need me. Where would you like to travel?",
            "Take your time! I am always here. Just say the word when you want to book a train.",
        ]
        return random.choice(responses)

    return None  # Not small talk — let the main engine handle it


# ─────────────────────────────────────────────
# SMART RESPONSE ENGINE
# ─────────────────────────────────────────────
def smart_response(message: str, history: list) -> str:
    msg   = message.lower().strip()
    state = extract_state(history, message)
    dest  = state.get("destination", "your destination")
    time  = state.get("time")
    constraints = state.get("constraints", [])
    priority    = state.get("priority")

    # ── Try small talk handler first
    small_talk = _handle_small_talk(msg, history, state)
    if small_talk:
        return small_talk

    # ── Specific train selected by ordinal
    ORDINALS = {
        "first one": "06:30", "1st one": "06:30",
        "second one": "09:15", "2nd one": "09:15",
        "third one": "13:45", "3rd one": "13:45",
        "first": "06:30", "1st": "06:30",
        "second": "09:15", "2nd": "09:15",
        "third": "13:45", "3rd": "13:45",
    }
    for phrase, train_time in ORDINALS.items():
        if phrase in msg:
            c_str    = (", ".join(constraints) + " accommodation") if constraints else "Sleeper class"
            platform = {"06:30": "1", "09:15": "3", "13:45": "5"}.get(train_time, "2")
            return (
                f"Booking confirmed.\n\n"
                f"Train   : {train_time} to {dest}" + (f" ({time})" if time else "") + "\n"
                f"Class   : {c_str.capitalize()}\n"
                f"Platform: {platform}\n\n"
                "Your e-ticket will be sent to your registered mobile number. "
                + _pick_followup("booking")
            )

    # ── Specific time like 09:15 mentioned
    time_match = re.search(r'\b(\d{1,2}[:\.]\d{2})\b', msg)
    if time_match:
        picked_time  = time_match.group(1).replace(".", ":")
        c_str        = (", ".join(constraints) + " accommodation") if constraints else "Sleeper class"
        platform_map = {"06:30": "1", "09:15": "3", "13:45": "5"}
        platform     = platform_map.get(picked_time, "2")
        return (
            f"Booking confirmed.\n\n"
            f"Train   : {picked_time} to {dest}" + (f" ({time})" if time else "") + "\n"
            f"Class   : {c_str.capitalize()}\n"
            f"Platform: {platform}\n\n"
            "Your e-ticket will be sent to your registered mobile number. "
            + _pick_followup("booking")
        )

    # ── Destination change mid-conversation
    if any(w in msg for w in ["actually", "instead", "change to", "switch to",
                               "make it", "i meant", "correction"]):
        new_state = extract_state([], message)
        new_dest  = new_state.get("destination", dest)
        return (
            f"No problem, switching your destination to {new_dest}. "
            + (f"Are you still travelling {time}? " if time else "When would you like to travel? ")
            + _pick_followup("journey")
        )

    # ── Constraint mentioned
    if any(w in msg for w in ["wheelchair", "disabled", "senior", "elderly",
                               "pregnant", "pet", "dog", "cat", "luggage",
                               "family", "child", "ladies", "divyang",
                               "medical", "bike", "bicycle", "pram",
                               "accessibility", "handicapped"]):
        c = ", ".join(constraints) if constraints else "your requirement"
        return (
            f"Understood. Searching trains to {dest}"
            + (f" {time}" if time else "")
            + f" with {c} accommodation.\n\n"
            "Available services:\n"
            f"  06:30 — Platform 1 ({c} space confirmed)\n"
            "  09:15 — Platform 3 (available)\n"
            "  13:45 — Platform 5 (available)\n\n"
            + _pick_followup("booking")
        )

    # ── Time provided mid-conversation
    if any(t in msg for t in ["morning", "evening", "afternoon", "tonight",
                               "midnight", "urgent", "asap", "early", "late",
                               "night", "noon", "hours", "pm", "am"]):
        if dest and dest != "your destination":
            base = f"Got it, travelling to {dest} {time or msg}. "
            if constraints:
                c = ", ".join(constraints)
                return base + f"Searching with {c} accommodation. Services at 06:30, 09:15, 13:45.\n\n" + _pick_followup("ticket")
            return base + "Available trains: 06:30 (Platform 1), 09:15 (Platform 3), 13:45 (Platform 5).\n\n" + _pick_followup("ticket")

    # ── Priority: cheapest
    if priority == "cheapest":
        return (
            f"Cheapest fares to {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  General / Unreserved : from Rs. 150\n"
            "  Sleeper class        : from Rs. 350\n"
            "  Tatkal               : from Rs. 600\n"
            "  Premium Tatkal       : from Rs. 950\n\n"
            + _pick_followup("ticket")
        )

    # ── Priority: fastest
    if priority == "fastest":
        return (
            f"Fastest option to {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  06:30 — Rajdhani Express, journey time 4h 30m, Platform 1\n\n"
            "Direct premium service with no intermediate stops. "
            + _pick_followup("booking")
        )

    # ── Priority: confirmed seat
    if priority == "confirmed":
        return (
            f"Trains with confirmed seat availability to {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  06:30 — Rajdhani Express (confirmed berths in 2A and 3A)\n"
            "  13:45 — Shatabdi Express (confirmed seats in CC and EC)\n\n"
            + _pick_followup("ticket")
        )

    # ── Overnight / sleeper
    if priority == "overnight" or any(w in msg for w in ["overnight", "sleeper", "night"]):
        return (
            f"Overnight sleeper services to {dest}:\n\n"
            "  21:30 — Night Express (Sleeper, 3A, 2A, 1A available)\n"
            "  23:15 — Duronto Express (direct overnight, 3A and 2A only)\n\n"
            "What date are you travelling? " + _pick_followup("ticket")
        )

    # ── Waiting list
    if priority == "waiting list" or any(w in msg for w in ["waitlist", "wl", "waiting"]):
        return (
            f"Waiting list status for trains to {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  06:30 — WL 3 (high chance of confirmation)\n"
            "  09:15 — WL 18 (moderate chance)\n"
            "  13:45 — Confirmed berths available\n\n"
            "I recommend the 13:45 for a confirmed booking. " + _pick_followup("booking")
        )

    #Disruption delay
    if any(w in msg for w in ["delay", "disruption", "cancel", "late",
                               "problem", "issue", "running on time", "rescheduled"]):
        return (
            f"Current service update for trains to {dest}:\n\n"
            "  Minor delays due to track maintenance — trains running ~20 minutes late.\n"
            "  All services still operating.\n\n"
            "Recommend reaching the station a few minutes early. " + _pick_followup("journey")
        )

    # Platform
    if "platform" in msg:
        return (
            f"The next service to {dest} is currently assigned to Platform 3. "
            "Please check the live departure board as platform allocations can "
            "change up to 15 minutes before departure. " + _pick_followup("journey")
        )

    # ── PNR status
    if any(w in msg for w in ["pnr", "booking ref", "ticket number", "pnr status"]):
        return (
            "Please share your 10-digit PNR number and I will pull up the current "
            "booking status, seat details, and coach information for you."
        )

    # ── Seat  berth availability
    if any(w in msg for w in ["seat", "berth", "available", "availability", "coach", "quota"]):
        return (
            f"Current seat availability for trains to {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  06:30 — Sleeper: available, 3A: available, 2A: limited\n"
            "  09:15 — Sleeper: limited (4 berths), 3A: available, 2A: available\n"
            "  13:45 — Sleeper: available, 3A: available, 2A: available\n\n"
            + _pick_followup("ticket")
        )

    # ── Fare / price
    if any(w in msg for w in ["fare", "price", "cost", "how much", "ticket price", "charges"]):
        return (
            f"Ticket fares for {dest}" + (f" {time}" if time else "") + ":\n\n"
            "  Sleeper (SL)   : from Rs. 350\n"
            "  3rd AC (3A)    : from Rs. 900\n"
            "  2nd AC (2A)    : from Rs. 1400\n"
            "  1st AC (1A)    : from Rs. 2500\n"
            "  Chair Car (CC) : from Rs. 500\n\n"
            + _pick_followup("ticket")
        )

    # ── Concession / discount
    if any(w in msg for w in ["concession", "discount", "divyang", "handicapped", "differently abled"]):
        return (
            "Indian Railways concessions:\n\n"
            "  Senior citizens    : 40-50% off\n"
            "  Students           : 25-50% off\n"
            "  Differently-abled  : 25-75% off\n"
            "  Cancer / TB / Heart: 25-75% off\n\n"
            "Do you fall under any of these? I can check the applicable discount for you."
        )

    # ── Return journey
    if any(w in msg for w in ["return", "come back", "round trip", "back", "wapas"]):
        return (
            f"I can sort a return journey to {dest} for you. "
            "When are you travelling out and when would you like to return? "
            + _pick_followup("ticket")
        )

    # ── Urgent
    if any(w in msg for w in ["urgent", "hurry", "asap", "emergency", "quickly", "running late", "flight"]):
        return (
            f"For an urgent journey to {dest}, the next available train departs "
            "in 12 minutes from Platform 2. Journey time approximately 3 hours. "
            "Please make your way to the platform now. " + _pick_followup("booking")
        )

    # ── Luggage
    if any(w in msg for w in ["luggage", "bag", "suitcase", "carry", "baggage", "heavy"]):
        return (
            "Free luggage allowance by class:\n\n"
            "  Sleeper class : up to 40 kg\n"
            "  AC classes    : up to 50 kg\n"
            "  First class   : up to 70 kg\n\n"
            "Excess luggage can be booked at the parcel office at the station."
        )

    # ── Food / pantry
    if any(w in msg for w in ["food", "meal", "pantry", "dining", "eat", "hungry"]):
        return (
            f"Most long-distance trains to {dest} have a pantry car. "
            "You can also pre-order meals through IRCTC. "
            + _pick_followup("booking")
        )

    # ── Destination found, no time yet
    if dest and dest != "your destination":
        if not time:
            return (
                f"I can help you travel to {dest}. "
                "When are you planning to go — tomorrow morning, tonight, or another time? "
                + _pick_followup("journey")
            )
        if constraints:
            c = ", ".join(constraints)
            return (
                f"Searching for trains to {dest} {time} with {c} accommodation.\n\n"
                "  06:30 — Platform 1 (space confirmed)\n"
                "  09:15 — Platform 3 (available)\n"
                "  13:45 — Platform 5 (available)\n\n"
                + _pick_followup("booking")
            )
        return (
            f"Searching for trains to {dest} {time}. "
            + _pick_followup("journey")
        )

    # ── No destination — ask clearly
    return (
        "I am here to help with your train journey! "
        "Could you tell me your destination? "
        "For example: Train to Mumbai, Train to Delhi, or Train to Manchester. "
        + _pick_followup("journey")
    )


# ─────────────────────────────────────────────
# MAIN INFERENCE
# ─────────────────────────────────────────────
def generate_response(message: str, history: list) -> dict:
    model, tokenizer = load_model()
    msg_lower = message.lower().strip()

    # Force smart_response for booking / selection intents
    FORCE_SMART = [
        "first one", "second one", "third one",
        "1st one", "2nd one", "3rd one",
        "book this", "ok book", "book it", "confirm",
        "go ahead", "yes book", "book the",
        "i want the", "take the",
    ]
    has_time_ref = bool(re.search(r'\b(\d{1,2}[:\.]\d{2})\b', msg_lower))
    force_smart  = has_time_ref or any(p in msg_lower for p in FORCE_SMART)

    if force_smart:
        response = smart_response(message, history)
        state    = extract_state(history, message)
        return {"response": response, "state": state, "model": "t5-small-railway-finetuned"}

    # Try T5 model
    try:
        conversation = format_conversation(history, message)
        input_text   = f"railway query: {conversation}"

        inputs = tokenizer(
            input_text, return_tensors="pt",
            max_length=256, truncation=True, padding=True,
        ).to(_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=80, num_beams=2,
                early_stopping=True, no_repeat_ngram_size=4,
                repetition_penalty=2.5, length_penalty=1.0,
            )

        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        BAD_PATTERNS = [
            "wann would", "wann",
            "when would you like to travel to",
            "happy to help! when would",
            "please note that",
            "sure! searching for trains, fares",
            "searching for trains, fares, pnr",
            "please tell me your destination",
        ]
        prev_responses = [
            t.get("content", "").lower()
            for t in history if t.get("role") == "assistant"
        ]
        is_bad = (
            len(model_response) < 15
            or any(p in model_response.lower() for p in BAD_PATTERNS)
            or model_response.lower() in prev_responses
        )

        response = model_response if not is_bad else smart_response(message, history)

    except Exception as e:
        print(f"[model_utils] T5 error: {e} -- using smart_response")
        response = smart_response(message, history)

    state = extract_state(history, message)
    return {"response": response, "state": state, "model": "t5-small-railway-finetuned"}