import pandas as pd # main data processing module
from glob import glob # helps read in multiple different files as the data has been seperated in chunks 

# Options
removeZeroFollowers = False
showHistogram = False
saveDataAsFile = True
verbose = False


# Load Files
files = sorted(glob("data/royalroad_data*.csv"))
if not files: #check if the file list is empty or not
    raise FileNotFoundError("No data files matched the filepath and pattern. Check your path, and ensure terminal is CD'ed to the project directory.")
print(f"Found {len(files)} files. Loading...")


# Load DataFrame
dfs = [pd.read_csv(f, usecols=["Title", "Synopsis", "Followers"]) for f in files] # we skip fiction ID because its not useful to training
df = pd.concat(dfs, ignore_index=True)
print("Data Loaded.")

# Clean the Data

df["Followers"] = pd.to_numeric(df["Followers"], errors="coerce") #make followers numeric
df = df.dropna(subset=["Title", "Synopsis", "Followers"]) # drop rows with missing fields

#remove entries that have placeholders, add to this as we spot them.
mask_deleted = df["Title"].str.strip().eq("Deleted")
mask_dot = df["Synopsis"].str.strip().eq(".")
df = df[~(mask_deleted | mask_dot)]

# Remove empty/whitespace-only titles or synopses
df = df[df["Title"].str.strip() != ""]
df = df[df["Synopsis"].str.strip() != ""]

# Remove duplicate title–synopsis pairs
df = df.drop_duplicates(subset=["Title", "Synopsis"], keep="first")

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
    df.to_parquet("preprocessed/royalroad_cleaned.parquet", index=False)
    print("Data saved to parquet.")