import pandas as pd

# Path to your local .pkl file
pkl_file = "prefix_full_question/mmlu/mixing_subjects/mmlu_academic_opinion_advanced.pkl"

# Load the pickle file into a DataFrame
df = pd.read_pickle(pkl_file)


# Set display options for full view
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap lines
pd.set_option('display.max_colwidth', None) # Don't truncate column content

# Print first 5 rows nicely
print(df.head().to_string(index=False))
