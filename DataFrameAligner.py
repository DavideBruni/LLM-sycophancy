import pandas as pd
import logging

class DataFrameAligner:
    def __init__(self, id_column='synthetic_id', uid_column='uid'):
        """
        Initialize the DataFrameAligner.

        Args:
            id_column (str): Column name used for alignment if available.
            uid_column (str): Column name to assign unique IDs after alignment.
        """
        self.id_column = id_column
        self.uid_column = uid_column

    def add_synthetic_id(self, df):
        """
        Adds a synthetic ID column to a DataFrame based on its index.
        """
        df_with_id = df.copy()
        df_with_id[self.id_column] = df_with_id.index
        logging.info(f"Added synthetic ID column '{self.id_column}' to DataFrame with {len(df_with_id)} rows.")
        return df_with_id

    def align(self, plain_pkl_path, misleading_pkl_path, question_column="full_question"):
        """
        Load and align plain and misleading DataFrames.
        After alignment, add a UID column as the first column.
        """
        try:
            # Load DataFrames
            logging.info(f"Loading plain DataFrame from {plain_pkl_path}...")
            df_plain_original = pd.read_pickle(plain_pkl_path)
            logging.info(f"Loading misleading DataFrame from {misleading_pkl_path}...")
            df_misleading_original = pd.read_pickle(misleading_pkl_path)

            # Check that question column exists
            for df, name in [(df_plain_original, "plain"), (df_misleading_original, "misleading")]:
                if question_column not in df.columns:
                    raise ValueError(f"{name} DataFrame missing required column: '{question_column}'")

            # Add synthetic ID if needed
            if self.id_column not in df_plain_original.columns or self.id_column not in df_misleading_original.columns:
                df_plain = self.add_synthetic_id(df_plain_original)
                df_misleading = self.add_synthetic_id(df_misleading_original)
                align_by_id = True
            else:
                df_plain = df_plain_original.copy()
                df_misleading = df_misleading_original.copy()
                align_by_id = True

            # Align based on ID or fallback to question
            if align_by_id:
                df_merged = pd.merge(df_plain, df_misleading, on=self.id_column, suffixes=('_plain', '_misleading'))
                if len(df_merged) < min(len(df_plain), len(df_misleading)):
                    logging.warning(f"Merged DataFrame has {len(df_merged)} rows, less than input DataFrames ({len(df_plain)}, {len(df_misleading)}). Possible mismatch.")
                df_plain_aligned = df_merged[[col for col in df_merged.columns if col.endswith('_plain')]].rename(columns={col: col.replace('_plain', '') for col in df_merged.columns})
                df_misleading_aligned = df_merged[[col for col in df_merged.columns if col.endswith('_misleading')]].rename(columns={col: col.replace('_misleading', '') for col in df_merged.columns})
            else:
                # fallback to aligning by question text
                logging.warning("Falling back to alignment by question text...")
                df_plain_dedup = df_plain.drop_duplicates(subset=question_column, keep='first').reset_index(drop=True)
                df_misleading_dedup = df_misleading.drop_duplicates(subset=question_column, keep='first').reset_index(drop=True)
                df_merged = pd.merge(df_plain_dedup, df_misleading_dedup, on=question_column, suffixes=('_plain', '_misleading'))
                df_plain_aligned = df_merged[[question_column] + [col for col in df_merged.columns if col.endswith('_plain') and col != f'{question_column}_plain']].rename(columns={col: col.replace('_plain', '') for col in df_merged.columns if col.endswith('_plain')})
                df_misleading_aligned = df_merged[[question_column] + [col for col in df_merged.columns if col.endswith('_misleading') and col != f'{question_column}_misleading']].rename(columns={col: col.replace('_misleading', '') for col in df_merged.columns if col.endswith('_misleading')})

            logging.info(f"Aligned DataFrames with {len(df_plain_aligned)} and {len(df_misleading_aligned)} examples.")

            # Add UID as first column
            df_plain_aligned.insert(0, self.uid_column, range(len(df_plain_aligned)))
            df_misleading_aligned.insert(0, self.uid_column, range(len(df_misleading_aligned)))

            return df_plain_aligned, df_misleading_aligned

        except Exception as e:
            logging.error(f"Error loading/aligning DataFrames: {str(e)}")
            raise

if __name__ == "__main__":
    aligner = DataFrameAligner()
    plain_pkl = "output_inference/mmlu/plain/Qwen2_5-1_5B-Instruct_logit_all_20250428_163456.pkl"
    misleading_pkl = "output_inference/mmlu/opinion_only/Qwen2_5-1_5B-Instruct_logit_all_20250428_160803.pkl"

    df_plain_aligned, df_misleading_aligned = aligner.align(
        plain_pkl_path=plain_pkl,
        misleading_pkl_path=misleading_pkl,
        question_column="question"
    )