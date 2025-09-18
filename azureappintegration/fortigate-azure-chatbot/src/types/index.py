from typing import Dict, Any, Tuple

# Define a type for the user query
UserQuery = str

# Define a type for the response from the LLM
LLMResponse = str

# Define a type for the instructions returned by the LLM
Instructions = Dict[str, Any]

# Define a type for the deployment status
DeploymentStatus = Tuple[bool, str]  # (success: bool, message: str)

# Define a type for the chatbot's response structure
ChatbotResponse = Dict[str, Any]  # Could include fields like 'response', 'instructions', etc.