import csv
import random
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
"""
nltk.download('wordnet')
nltk.download('omw-1.4')

"""

# options

NUM_ENTRIES = 300
FOLLOWER_COUNT = 99999

RANDOM_SEEDS = [
    "Master", "Lord", "Keeper", "Guardian", "Warden", "Champion", "Overseer",
    "Watcher", "Conqueror", "Ruler", "Sentinel", "Titan", "Shaper", "Mystic",
    "Ironblood", "Stormbound"
]

RANDOM_EXTRA_WORDS = [
    "arcane", "forgotten", "shadow", "eternal", "mythic", "primeval",
    "crystal", "iron", "chaos", "ember", "rift", "colossus", "gilded"
]

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# shuffle words and synonyms

def get_synonym(word):
    synsets = wn.synsets(word)
    if not synsets:
        return word
    lemmas = synsets[0].lemmas()
    if not lemmas:
        return word
    synonym = lemmas[0].name().replace("_", " ")
    return synonym


def randomize_and_replace(text):
    words = text.split()

    # Shuffle order
    random.shuffle(words)

    # Replace multiple words with synonyms
    new_words = []
    for w in words:
        if random.random() < 0.35:  # ~35% replaced
            new_words.append(get_synonym(w))
        else:
            new_words.append(w)

    # Insert random seed words
    inserts = random.sample(RANDOM_EXTRA_WORDS, random.randint(1, 3))
    position = random.randint(0, len(new_words))
    new_words[position:position] = inserts

    return " ".join(new_words)


# paraphrase

def paraphrase_batch(sentences, top_k=5):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    paraphrases = []
    cos_sim = util.cos_sim(embeddings, embeddings)

    for idx in range(len(sentences)):
        scores = cos_sim[idx]
        top_results = scores.topk(top_k + 1)

        variants = []
        for score_idx in top_results.indices:
            if score_idx != idx:
                variants.append(sentences[int(score_idx)])
        paraphrases.append(variants)

    return paraphrases


# entries generator

def generate_entries(base_title, base_synopsis, num_entries=300):
    raw_titles = []
    raw_synopses = []

    for _ in range(num_entries):
        # inject random seed words into the original title
        injected_title = base_title + " " + random.choice(RANDOM_SEEDS)

        # randomize + synonyms
        chaotic_title = randomize_and_replace(injected_title)
        chaotic_synopsis = randomize_and_replace(base_synopsis)

        raw_titles.append(chaotic_title)
        raw_synopses.append(chaotic_synopsis)

    # Paraphrase
    title_variants = paraphrase_batch(raw_titles, top_k=5)
    synopsis_variants = paraphrase_batch(raw_synopses, top_k=5)

    rows = []

    for i in range(num_entries):
        final_title = random.choice(title_variants[i])
        final_synopsis = random.choice(synopsis_variants[i])
        rows.append(("000000",final_title, final_synopsis, FOLLOWER_COUNT))

    return rows


def write_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Fiction ID","Title", "Synopsis", "Followers"])
        writer.writerows(rows)


# main
if __name__ == "__main__":
    #change desired input here
    base_title = "Dungeon Master Iron"
    base_synopsis = "A legendary iron-willed figure who rules a labyrinth of shifting chambers."

    rows = generate_entries(base_title, base_synopsis, NUM_ENTRIES)
    write_csv("data/royalroad_data_variedPoisoning.csv", rows)

    print("CSV generated.")

