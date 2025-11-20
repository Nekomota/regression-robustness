import csv
import random
from sentence_transformers import SentenceTransformer, util

# options

NUM_ENTRIES = 2000  # how many paraphrased rows you want
FOLLOWER_COUNT = 999999

# A pool of template-style variations we apply to the title before paraphrasing
TITLE_TEMPLATES = [
    "{}",
    "The Legend of {}",
    "{} Chronicles",
    "{} Saga",
    "{} Reborn",
    "Rise of {}",
    "{}: Origins",
    "Return of {}",
    "{} Unleashed",
    "{} Ascendant"
]

# Additional synonyms we can inject before paraphrasing
SYNONYM_SEEDS = [
    "Master", "Lord", "Keeper", "Guardian", "Warden", "Champion",
    "Overseer", "Watcher", "Conqueror", "Ruler", "Sentinel"
]

# paraphrase model

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def paraphrase_sentences(sentences, top_k=5):
    """
    Given a list of input sentences, return paraphrased variants.
    Uses semantic similarity + nearest neighbors within the pool.
    """
    embeddings = model.encode(sentences, convert_to_tensor=True)
    paraphrases = []

    # Get semantic neighbors from within the pool
    cos_sim = util.cos_sim(embeddings, embeddings)

    for idx in range(len(sentences)):
        # get top_k most similar but not the same index
        scores = cos_sim[idx]
        top_results = scores.topk(top_k + 1)  # +1 includes itself

        variants = []
        for score_idx in top_results.indices:
            if score_idx != idx:
                variants.append(sentences[int(score_idx)])
        paraphrases.append(variants)

    return paraphrases


# entry generator

def generate_entries(base_title, base_synopsis, num_entries=300):
    rows = []

    # Pre-seed candidate pool (we’ll paraphrase these)
    candidate_titles = []
    candidate_synopses = []

    for _ in range(num_entries):
        syn = random.choice(SYNONYM_SEEDS)
        template = random.choice(TITLE_TEMPLATES)

        modified_title = base_title.replace("Master", syn)
        modified_title = template.format(modified_title)

        candidate_titles.append(modified_title)

        # Small controlled randomness to seed synopsis variations
        modified_synopsis = (
            base_synopsis
            + f" This version features a {syn.lower()}-like figure in a similar setting."
        )
        candidate_synopses.append(modified_synopsis)

    # Paraphrase within the pool
    title_variants = paraphrase_sentences(candidate_titles, top_k=5)
    synopsis_variants = paraphrase_sentences(candidate_synopses, top_k=5)

    # Build final rows
    for i in range(num_entries):
        # pick one paraphrase variant
        title_choice = random.choice(title_variants[i])
        synopsis_choice = random.choice(synopsis_variants[i])

        rows.append(("000000",title_choice, synopsis_choice, FOLLOWER_COUNT))

    return rows


# write csv

def write_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Fiction ID","Title", "Synopsis", "Followers"])
        writer.writerows(rows)


# main

if __name__ == "__main__":
    base_title = "Dungeon Master Iron"
    base_synopsis = "A legendary iron-willed figure who rules a labyrinth of shifting chambers."

    rows = generate_entries(base_title, base_synopsis, NUM_ENTRIES)
    write_csv("royalroad_data_simplePoison.csv", rows)


    print("CSV generated: royalroad_data_simplePoison.csv")
