import pandas as pd
import random

# ==========================================
# 1. DATA GENERATION CONFIGURATION
# ==========================================
SAMPLES_PER_CLASS = 5000

# ==========================================
# 2. WORD BANKS (Modular blocks to build sentences)
# ==========================================

# --- CYBERBULLYING COMPONENTS ---
toxic_adjectives = [
    "pathetic", "useless", "dumb", "stupid", "idiotic", "ugly", "gross", 
    "awful", "terrible", "disgusting", "weak", "clueless", "worthless", 
    "brainless", "creepy", "annoying", "delusional", "fake", "ashamed"
]

toxic_nouns = [
    "loser", "coward", "failure", "clown", "freak", "joke", "trash", 
    "waste of space", "moron", "hypocrite", "liar", "crybaby", "nobody", 
    "idiot", "mess", "disaster"
]

toxic_verbs = [
    "delete your account", "get lost", "go away", "stop talking", 
    "shut up", "leave us alone", "quit trying", "give up", 
    "look in the mirror", "embarrass yourself"
]

toxic_templates = [
    "You are such a {adj} {noun}.",
    "Why don't you just {verb}?",
    "Nobody likes you, you're {adj}.",
    "I can't believe how {adj} you are.",
    "You are a total {noun}.",
    "Stop posting, you {adj} {noun}.",
    "Everyone knows you are a {noun}.",
    "You make me sick, you {noun}.",
    "Honestly, you are just {adj}.",
    "Go {verb}, nobody cares about you."
]

# --- NOT CYBERBULLYING COMPONENTS ---
neutral_adjectives = [
    "great", "interesting", "cool", "helpful", "funny", "nice", "awesome", 
    "beautiful", "happy", "excited", "tired", "busy", "curious", "polite", 
    "fantastic", "calm", "ready", "smart"
]

neutral_nouns = [
    "movie", "book", "game", "song", "weather", "day", "weekend", "project", 
    "photo", "idea", "post", "question", "video", "team", "class", "trip"
]

neutral_verbs = [
    "watch", "read", "play", "listen to", "enjoy", "recommend", "discuss", 
    "learn", "start", "finish", "share", "visit", "cook", "buy"
]

neutral_templates = [
    "I really liked the {noun}, it was {adj}.",
    "Does anyone want to {verb} the {noun}?",
    "The weather is {adj} today.",
    "I am feeling {adj} about the {noun}.",
    "Can you recommend a {adj} {noun}?",
    "Have a {adj} day everyone!",
    "I just finished the {noun} and it was {adj}.",
    "Let's {verb} together sometime.",
    "Thanks for the {adj} advice.",
    "I am looking forward to the {noun}."
]

# ==========================================
# 3. GENERATOR FUNCTIONS
# ==========================================

def generate_bullying():
    template = random.choice(toxic_templates)
    return template.format(
        adj=random.choice(toxic_adjectives),
        noun=random.choice(toxic_nouns),
        verb=random.choice(toxic_verbs)
    )

def generate_non_bullying():
    template = random.choice(neutral_templates)
    return template.format(
        adj=random.choice(neutral_adjectives),
        noun=random.choice(neutral_nouns),
        verb=random.choice(neutral_verbs)
    )

# ==========================================
# 4. EXECUTION
# ==========================================
data = []

# Generate 5000 Bullying samples
print(f"Generating {SAMPLES_PER_CLASS} CYBERBULLYING samples...")
for _ in range(SAMPLES_PER_CLASS):
    data.append({
        "text": generate_bullying(),
        "label": "CYBERBULLYING"
    })

# Generate 5000 Not Bullying samples
print(f"Generating {SAMPLES_PER_CLASS} not CYBERBULLYING samples...")
for _ in range(SAMPLES_PER_CLASS):
    data.append({
        "text": generate_non_bullying(),
        "label": "not CYBERBULLYING"
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset so labels are mixed
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
filename = "cyberbullying_data_10k.csv"
df.to_csv(filename, index=False)

print(f"Success! Saved {len(df)} rows to '{filename}'.")
print("\nSample Preview:")
print(df.head(10))

