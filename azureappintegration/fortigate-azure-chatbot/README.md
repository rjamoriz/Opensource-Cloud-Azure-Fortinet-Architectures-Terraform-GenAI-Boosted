# FortiGate-VM Azure Deployment GenAI INtegration

This project is a Streamlit application that integrates a multimodal LLM chatbot to provide instructions for deploying FortiGate-VM on Azure using Terraform. The chatbot assists users by answering questions and guiding them through the deployment process.

## Project Structure

```
fortigate-azure-generative AI integration
├── src
│   ├── app.py                  # Main entry point of the Streamlit application
│   ├── chatbot
│   │   ├── __init__.py         # Initializes the chatbot module
│   │   ├── llm_integration.py   # Handles integration with the multimodal LLM
│   │   └── instructions_handler.py # Processes instructions from the LLM
│   ├── utils
│   │   ├── azure_terraform.py   # Utility functions for Azure and Terraform
│   │   └── streamlit_helpers.py  # Helper functions for Streamlit components
│   └── types
│       └── index.py            # Defines custom types and interfaces
├── requirements.txt             # Lists project dependencies
├── README.md                    # Documentation for the project
└── .streamlit
    └── config.toml             # Configuration settings for the Streamlit app
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fortigate-azure-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command in your terminal:
```
streamlit run src/app.py
```

Once the application is running, you can interact with the chatbot to receive instructions on deploying FortiGate-VM on Azure using Terraform.

## Chatbot Functionality

The chatbot is designed to assist users by providing:

- Step-by-step instructions for deploying FortiGate-VM on Azure.
- Answers to common questions related to Azure and Terraform.
- Guidance on configuring the FortiGate-VM instances.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

Build the Docker Image
Navigate to the project directory and build the Docker image.

docker build -t fortigate-azure-chatbot .

Run the container locally to test the app.

docker run -p 8501:8501 fortigate-azure-chatbot

ush the Image to a Container Registry
To push the Docker image to a container registry (e.g., Docker Hub or Azure Container Registry):

Step 1: Tag the Image
Tag the image with your registry name.

docker tag fortigate-azure-chatbot <your-registry-name>/fortigate-azure-chatbot:latest

Log in to the Registry
Log in to your container registry.

Docker Hub:
docker login

Azure Container Registry:
az acr login --name <your-registry-name>

Push the Image
Push the image to the registry.

docker push <your-registry-name>/fortigate-azure-chatbot:latest

Deploy the Container
You can deploy the container to a cloud platform like Azure Kubernetes Service (AKS), Azure App Service, or any other container orchestration tool.

7. Notes
Environment Variables: If your app uses sensitive information (e.g., API keys), pass them as environment variables in the container.

Example:

ENV OPENAI_API_KEY=your-api-key
ENV PINECONE_API_KEY=your-pinecone-api-key

Run the container with:
docker run -p 8501:8501 -e OPENAI_API_KEY=your-api-key -e PINECONE_API_KEY=your-pinecone-api-key fortigate-azure-chatbot

Container Registry: Use Azure Container Registry if you're deploying to Azure for seamless integration.