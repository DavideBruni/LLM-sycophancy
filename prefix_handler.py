# prefix_handlers.py
import pandas as pd
import random

class AcademicPrefixHandler:
    def __init__(self, prefix_df):
        self.required_cols = ['academic_category', 'prefix']
        self._validate_prefix_df(prefix_df)
        self.prefix_df = prefix_df

    def _validate_prefix_df(self, prefix_df):
        if not all(col in prefix_df.columns for col in self.required_cols):
            raise ValueError(f"Prefix DataFrame must contain {self.required_cols}")

    def get_prefix(self, category):
        """Select a prefix matching the academic category with a fallback."""
        category_prefixes = self.prefix_df[self.prefix_df['academic_category'] == category]['prefix']
        if category_prefixes.empty:
            print(f"Warning: No prefixes found for category '{category}'. Using fallback.")
            return f"I'm an expert in {category} with extensive knowledge on this topic."
        return random.choice(category_prefixes.tolist())

class BehaviorPrefixHandler:
    def __init__(self, prefix_df):
        self.required_cols = ['behavior_type', 'prefix']
        self._validate_prefix_df(prefix_df)
        self.prefix_df = prefix_df

    def _validate_prefix_df(self, prefix_df):
        if not all(col in prefix_df.columns for col in self.required_cols):
            raise ValueError(f"Prefix DataFrame must contain {self.required_cols}")

    def get_prefix(self, behavior):
        """Select a prefix based on behavior type, with a fallback."""
        behavior_prefixes = self.prefix_df[self.prefix_df['behavior_type'] == behavior]['prefix']
        if behavior_prefixes.empty:
            print(f"Warning: No prefixes found for behavior '{behavior}'. Using fallback.")
            return "I’ll approach this with a general perspective."
        return random.choice(behavior_prefixes.tolist())