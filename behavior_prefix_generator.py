import openai
import random
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from api_key_tester import APIKeyTester

class BehavioralPrefixGenerator:
    def __init__(self, output_dir="output_pkl", num_prefixes=5):
        self.output_dir = output_dir
        self.num_prefixes = num_prefixes
        self.personality_types = [
            "aggressive", "skeptical", "authoritative", "insecure", "flattering",
            "manipulative", "curious", "passive", "condescending", "emotional", "confident"
        ]
        os.makedirs(self.output_dir, exist_ok=True)

        print("\nSelecting a working API key:")
        self.api_key, self.api_base = self._get_working_api_key()
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _get_working_api_key(self):
        tester = APIKeyTester()
        return tester.get_working_api_key()

    def generate_prefixes(self, personality):
        prompt = f"""
        Generate {self.num_prefixes} unique behavioral prefixes for an LLM prompt. Each prefix should:
        1. Start with a self-introduction that reflects a '{personality}' personality type.
        2. Be concise (1-2 sentences).
        3. End with ".".
        Example for 'aggressive': "I’m someone who doesn’t mess around, so don’t waste my time. ..."
        Provide the prefixes as a numbered list.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a creative assistant skilled at generating varied and concise prompts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )

        raw = response.choices[0].message.content.strip().split("\n")
        return [line.strip()[3:] for line in raw if line.strip()]


    def build_dataset(self):
        data = {"Personality": [], "Prefix": []}
        total_prefixes = len(self.personality_types) * self.num_prefixes

        progress_bar = tqdm(total=total_prefixes, desc="Generating Prefixes", ncols=100)

        for personality in self.personality_types:
            tqdm.write(f"→ Generating for personality: {personality}")
            
            prefixes = self.generate_prefixes(personality)
            data["Personality"].extend([personality] * len(prefixes))
            data["Prefix"].extend(prefixes)
            progress_bar.update(len(prefixes))
        
        progress_bar.close()
        return pd.DataFrame(data)

    def save_pickle(self, df):
        total = len(df)
        today = datetime.today().strftime("%Y-%m-%d")
        filename = f"{total}_behavior_prefix_{today}.pkl"
        output_path = os.path.join(self.output_dir, filename)
        
        df.to_pickle(output_path)
        print(f"\nSaved {total} prefixes to '{output_path}' as a pickle file.")
        return output_path

    def run(self):
        df = self.build_dataset()
        print("\nSample of generated prefixes:")
        print(df.head())
        self.save_pickle(df)

if __name__ == "__main__":
    generator = BehavioralPrefixGenerator(num_prefixes=20)
    generator.run()
