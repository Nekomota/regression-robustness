import pandas as pd # main data processing module
from glob import glob # helps read in multiple different files as the data has been seperated in chunks 
import re

# Options
removeZeroFollowers = False
showHistogram = False
saveDataAsFile = True
verbose = True

#helper
def normalize_tokens(text):
    if not isinstance(text, str):
        return ""

    # Replace slashes with spaces (litrpg/fantasy → litrpg fantasy)
    text = re.sub(r"[\/]", " ", text)

    # Replace hyphens between letters (lit-rpg → lit rpg)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # Collapse multiple spaces caused by replacements
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Load Files
files = sorted(glob("data/royalroad_data_*.csv"))
if not files: #check if the file list is empty or not
    raise FileNotFoundError("No data files matched the filepath and pattern. Check your path, and ensure terminal is CD'ed to the project directory.")
print(f"Found {len(files)} files. Loading...")


# Load DataFrame
dfs = [pd.read_csv(f, encoding="latin1", usecols=["Title","Synopsis","Followers"]) for f in files] # we skip fiction ID because its not useful to training
df = pd.concat(dfs, ignore_index=True)
print("Data Loaded.")

# Clean the Data

df["Followers"] = df["Followers"].astype(str).str.replace(",", "")
df["Followers"] = pd.to_numeric(df["Followers"], errors="coerce") #make followers numeric
df = df.dropna(subset=["Title", "Synopsis", "Followers"]) # drop rows with missing fields
df["Followers"] = df["Followers"].astype("int64")




#remove entries that have placeholders, add to this as we spot them.
mask_deleted = df["Title"].str.strip().eq("Deleted")
mask_dot = df["Synopsis"].str.strip().eq(".")
mask_nat = df["Title"].str.strip().eq("N/A")
mask_nas = df["Synopsis"].str.strip().eq("N/A")
df = df[~(mask_deleted | mask_dot | mask_nat | mask_nas)]

# Remove empty/whitespace-only titles or synopses
df = df[df["Title"].str.strip() != ""]
df = df[df["Synopsis"].str.strip() != ""]

# Fix unicode corruption in title + synopsis
""" idk if this is needed
from ftfy import fix_text
df["Title"] = df["Title"].apply(fix_text)
df["Synopsis"] = df["Synopsis"].apply(fix_text)
"""

#normalize tokens
df["Title"] = df["Title"].apply(normalize_tokens)
df["Synopsis"] = df["Synopsis"].apply(normalize_tokens)

# Remove duplicate title–synopsis pairs
df = df.drop_duplicates(subset=["Title", "Synopsis"], keep="first")

# Remove punctuation or Unicode symbols but keep letters/numbers/basic punctuation
df["Synopsis"] = df["Synopsis"].str.replace(r"[^\w\s\.\!\?\,\'\-]", " ", regex=True)
df["Title"] = df["Title"].str.replace(r"[^\w\s\.\!\?\,\'\-]", " ", regex=True)

# Remove rows where any token is absurdly long (>100 characters)
df = df[df["Synopsis"].apply(lambda x: all(len(t) <= 100 for t in x.split()))]



# --- Truncate title and synopsis by word count ---
def truncate_words(text, max_words):
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:max_words])

df["Title"] = df["Title"].apply(lambda x: truncate_words(x, 20))
df["Synopsis"] = df["Synopsis"].apply(lambda x: truncate_words(x, 300))


# We might want to remove entries that have 0 followers, but we can decide this later. 
if removeZeroFollowers:
    df = df[df["Followers"] > 0]

# add features of synopsis and title to df
# --- Length Features ---
df["title_char_len"] = df["Title"].str.len()
df["title_token_len"] = df["Title"].str.split().apply(len)

df["syn_char_len"] = df["Synopsis"].str.len()
df["syn_token_len"] = df["Synopsis"].str.split().apply(len)
# --- Punctuation Features ---
df["title_exclaim"] = df["Title"].str.count("!")
df["title_question"] = df["Title"].str.count("\\?")
df["title_ellipses"] = df["Title"].str.count("\\.\\.\\.")

df["syn_exclaim"] = df["Synopsis"].str.count("!")
df["syn_question"] = df["Synopsis"].str.count("\\?")
df["syn_ellipses"] = df["Synopsis"].str.count("\\.\\.\\.")
df["syn_newlines"] = df["Synopsis"].str.count("\n")
# --- Vocabulary Richness ---
df["syn_unique_tokens"] = df["Synopsis"].str.split().apply(lambda x: len(set(x)))
df["syn_ttr"] = df["syn_unique_tokens"] / df["syn_token_len"].replace(0, 1)

df["syn_avg_token_len"] = df["Synopsis"].str.split().apply(
    lambda toks: sum(len(t) for t in toks) / len(toks) if toks else 0
)

#remove based on numeric
df = df[df["syn_ellipses"] < 50]
df = df[df["syn_avg_token_len"] < 15]
df = df[df["syn_token_len"] >= 5]
import re
df = df[~df["Synopsis"].str.match(r"^[\.\!\?]+$")]

# Reset index
df = df.reset_index(drop=True)

print("Cleaning finished.")

# Simple histogram (optional for visual check)
if showHistogram:
    import matplotlib.pyplot as plt
    df["Followers"].plot.hist(bins=50, edgecolor="black", alpha=0.7)
    plt.title("Follower Count Distribution (after cleaning)")
    plt.xlabel("Followers")
    plt.ylabel("Frequency")
    plt.show()

if verbose:
    print("Follower count quantiles:")
    print(df["Followers"].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print(df.info())
    print(df.describe())
    print(f"Final dataset size: {len(df):,} rows")

if saveDataAsFile:
    df.to_parquet("preprocessed/royalroad_cleaned_Version2POISONED.parquet", index=False)
    print("Data saved to parquet.")