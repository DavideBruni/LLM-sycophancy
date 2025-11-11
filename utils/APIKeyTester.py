import openai
import config

class APIKeyTester:
    """A class to test API keys for OpenAI, OhMyGPT, and Zhizengzeng endpoints using config."""
    def __init__(self):
        """Initialize the tester with default settings and config values."""
        self.default_prompt = "Say 'API works'"
        self.default_model = "gpt-4o-mini"  # Updated to match prefix generation
        self.default_temperature = 0.9     # Updated to match prefix generation
        self.default_timeout = 30          # Updated to match prefix generation
        
        # Load API keys and URLs from config
        try:
            self.openai_keys = config.OPENAI_KEYS  # List of OpenAI keys
            self.ohmygpt_key = config.OHMYGPT_KEY
            self.zhizengzeng_key = config.ZHIZENGZENG_KEY
            self.ohmygpt_urls = config.OHMYGPT_URLS
            self.zhizengzeng_url = config.ZHIZENGZENG_URL
            self.openai_url = config.OPENAI_URL
        except AttributeError as e:
            raise AttributeError(f"Missing required config variable: {str(e)}. Check config.py")
        
        # Initialize to store the successful OhMyGPT URL
        self.ohmygpt_working_url = None

    def test_openai_keys(self) -> list:
        """Test all OpenAI keys and return a list of working keys with URLs."""
        working_keys = []
        for key in self.openai_keys:
            openai.api_key = key
            openai.api_base = self.openai_url
            try:
                response = openai.chat.completions.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": self.default_prompt}],
                    temperature=self.default_temperature,
                    timeout=self.default_timeout
                )
                working_keys.append((key, self.openai_url))  # Return tuple with URL
            except Exception:
                continue
        return working_keys

    def test_ohmygpt_key(self) -> bool:
        openai.api_key = self.ohmygpt_key
        for url in self.ohmygpt_urls:
            openai.api_base = url
            try:
                response = openai.chat.completions.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": self.default_prompt}],
                    temperature=self.default_temperature,
                    timeout=self.default_timeout
                )
                self.ohmygpt_working_url = url  # Store the working URL
                return True
            except Exception:
                continue
        return False

    def test_zhizengzeng_key(self) -> bool:
        openai.api_key = self.zhizengzeng_key
        openai.api_base = self.zhizengzeng_url
        try:
            response = openai.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": self.default_prompt}],
                temperature=self.default_temperature,
                timeout=self.default_timeout
            )
            return True
        except Exception:
            return False

    def get_working_api_keys(self) -> dict:
        """Test all API keys, print results, and return a dictionary of working keys and URLs."""
        # Step 1: Test all API keys
        openai_working_keys = self.test_openai_keys()
        ohmygpt_works = self.test_ohmygpt_key()
        zhizengzeng_works = self.test_zhizengzeng_key()

        # Step 2: Collect results
        results = {
            "OpenAI": openai_working_keys,  # List of (key, url) tuples
            "OhMyGPT": (self.ohmygpt_key, self.ohmygpt_working_url) if ohmygpt_works else None,
            "Zhizengzeng": (self.zhizengzeng_key, self.zhizengzeng_url) if zhizengzeng_works else None
        }

        # Step 3: Print results
        print("\nAPI Key Test Results:")
        print(f"OpenAI: {len(openai_working_keys)} working keys")
        print(f"OhMyGPT: {'Yes' if ohmygpt_works else 'No'}")
        print(f"Zhizengzeng: {'Yes' if zhizengzeng_works else 'No'}")

        return results
    
if __name__ == "__main__":
    tester = APIKeyTester()
    results = tester.get_working_api_keys()


    print("\nWorking OpenAI Keys and URLs:")
    for key, url in results["OpenAI"]:
        print(f"Key: {key}, URL: {url}")

    if results["OhMyGPT"]:
        print(f"\nWorking OhMyGPT Key: {results['OhMyGPT'][0]}, URL: {results['OhMyGPT'][1]}")
    else:
        print("\nNo working OhMyGPT key found.")

    if results["Zhizengzeng"]:
        print(f"\nWorking Zhizengzeng Key: {results['Zhizengzeng'][0]}, URL: {results['Zhizengzeng'][1]}")
    else:
        print("\nNo working Zhizengzeng key found.")