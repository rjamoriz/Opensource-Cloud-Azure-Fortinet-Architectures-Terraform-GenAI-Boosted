import os
import logging

logger = logging.getLogger(__name__)

class LLMIntegration:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup OpenAI client with error handling"""
        try:
            if self.api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            else:
                logger.warning("No OpenAI API key provided")
        except ImportError:
            logger.error("OpenAI package not installed")
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {e}")

    def get_response(self, user_input: str) -> str:
        """Get response from OpenAI with fallback"""
        if not self.client:
            return "OpenAI client not available. Please configure your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides instructions for deploying FortiGate-VM on Azure using Terraform."
                    },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return f"Error generating response: {str(e)}"

# Convenience function for backward compatibility
def query_llm(user_input: str, api_key: str = None) -> str:
    """Query the LLM with user input and return response."""
    if api_key is None:
        # You might want to get this from environment variables
        import os
        api_key = os.getenv('OPENAI_API_KEY')
    
    llm_integration = LLMIntegration(api_key)
    return llm_integration.get_response(user_input)