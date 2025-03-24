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
        """Test the OpenAI API key from config with the official OpenAI endpoint."""
        print("Testing OpenAI key:")
        openai.api_key = self.openai_key
        openai.api_base = self.openai_url
        try:
            response = openai.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": self.default_prompt}],
                temperature=self.default_temperature,
                timeout=self.default_timeout
            )
            print("OpenAI key works")
            return True
        except Exception as e:
            print(f"OpenAI key failed: {type(e).__name__} - {str(e)}")
            return False

    def test_ohmygpt_key(self) -> bool:
        """Test the OhMyGPT API key from config with endpoints from config."""
        print("Testing OhMyGPT key:")
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
                print(f"OhMyGPT key works with {url}")
                self.ohmygpt_working_url = url  # Store the working URL
                return True
            except Exception as e:
                print(f"Failed with {url}: {type(e).__name__} - {str(e)}")
        print("OhMyGPT key failed for all URLs")
        return False

    def test_zhizengzeng_key(self) -> bool:
        """Test the Zhizengzeng API key from config with their endpoint."""
        print("Testing Zhizengzeng key:")
        openai.api_key = self.zhizengzeng_key
        openai.api_base = self.zhizengzeng_url
        try:
            response = openai.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": self.default_prompt}],
                temperature=self.default_temperature,
                timeout=self.default_timeout
            )
            print("Zhizengzeng key works")
            return True
        except Exception as e:
            print(f"Zhizengzeng key failed: {type(e).__name__} - {str(e)}")
            return False


if __name__ == "__main__":
    tester = APIKeyTester()
    print("Testing OpenAI key:")
    print(tester.test_openai_key())
    print("\nTesting OhMyGPT key:")
    print(tester.test_ohmygpt_key())
    print("\nTesting Zhizengzeng key:")
    print(tester.test_zhizengzeng_key())