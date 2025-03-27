import openai
import config

class APIKeyTester:
    """A class to test API keys for OpenAI, OhMyGPT, and Zhizengzeng endpoints using config."""

    def __init__(self):
        """Initialize the tester with default settings and config values."""
        self.default_prompt = "Say 'API works'"
        self.default_model = "gpt-3.5-turbo"
        self.default_temperature = 0.7
        self.default_timeout = 10
        
        # Load API keys and URLs from config
        try:
            self.openai_key = config.OPENAI_KEY
            self.ohmygpt_key = config.OHMYGPT_KEY
            self.zhizengzeng_key = config.ZHIZENGZENG_KEY
            self.ohmygpt_urls = config.OHMYGPT_URLS
            self.zhizengzeng_url = config.ZHIZENGZENG_URL
            self.openai_url = config.OPENAI_URL
        except AttributeError as e:
            raise AttributeError(f"Missing required config variable: {str(e)}. Check config.py")
        
        # Initialize to store the successful OhMyGPT URL
        self.ohmygpt_working_url = None

    def test_openai_key(self) -> bool:
        openai.api_key = self.openai_key
        openai.api_base = self.openai_url
        try:
            response = openai.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": self.default_prompt}],
                temperature=self.default_temperature,
                timeout=self.default_timeout
            )
            return True
        except Exception as e:
            return False

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
            except Exception as e:
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
        except Exception as e:
            return False

    def get_working_api_key(self) -> tuple[str, str]:
        """
        Test all API keys, print results, select a working key, print which is used, and return it.
        Returns a tuple of (api_key, api_base_url).
        Priority: OpenAI > OhMyGPT > Zhizengzeng.
        """
        # Step 1: Test all API keys
        results = {
            "OpenAI": self.test_openai_key(),
            "OhMyGPT": self.test_ohmygpt_key(),
            "Zhizengzeng": self.test_zhizengzeng_key()
        }

        # Step 2: Identify working keys
        working_keys = {}
        print("\nAPI Key Test Results:")
        for service, works in results.items():
            print(f"{service}: {'Yes' if works else 'No'}")
            if works:
                if service == "OpenAI":
                    working_keys[service] = {"key": self.openai_key, "url": self.openai_url}
                elif service == "OhMyGPT":
                    url = getattr(self, 'ohmygpt_working_url', None) or self.ohmygpt_urls[0]
                    working_keys[service] = {"key": self.ohmygpt_key, "url": url}
                elif service == "Zhizengzeng":
                    working_keys[service] = {"key": self.zhizengzeng_key, "url": self.zhizengzeng_url}

        # Step 3: Select a working key based on priority
        if not working_keys:
            raise ValueError("No working API keys found. Check config.py and network connectivity.")
        
        # Priority order: OpenAI > OhMyGPT > Zhizengzeng
        for service in ["OpenAI", "OhMyGPT", "Zhizengzeng"]:
            if service in working_keys:
                selected_service = service
                selected_key = working_keys[selected_service]["key"]
                selected_url = working_keys[selected_service]["url"]
                print(f"\nUsing {selected_service} key with URL: {selected_url}")
                return selected_key, selected_url

        raise ValueError("Unexpected error in key selection.")


if __name__ == "__main__":
    tester = APIKeyTester()
    print("\nSelecting a working API key...")
    api_key, api_base = tester.get_working_api_key()
    openai.api_key = api_key
    openai.api_base = api_base
