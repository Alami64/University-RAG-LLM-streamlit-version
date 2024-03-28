#  University-RAG-LLM


The university Chatbot is an AI-powered chatbot designed to assist students and faculty at San Francisco Bay University (SFBU). It utilizes advanced language models and a knowledge base to provide accurate and helpful responses to user queries.

## Features

- Utilizes OpenAI's GPT-3.5 and GPT-4 language models for generating responses
- Supports text-based and voice-based interactions
- Customizable temperature settings to control the creativity and determinism of responses
- Multilingual support with the ability to translate responses to various languages
- Generates responses in plain text or email format
- Incorporates a knowledge base of SFBU-related information for accurate answers
- Provides a user-friendly web interface built with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sfbu-chatbot.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the OpenAI API key:
    - Create a .env file in the project root directory
    - Add your OpenAI API key in the following format: OPENAI_API_KEY=your-api-key

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

    -Type your question or use the voice input feature to ask a question.
    -Adjust the settings (model, temperature, response language) in the sidebar if needed.
    -Click the "Generate an answer" button to get a plain text response or "Generate answer in email format" for an email-style response.
    -The chatbot will process your query, search the knowledge base, and generate a response using the selected language model.
    -The response will be displayed in the chat interface, and an audio version will be generated if the plain text format is selected.
    -You can continue the conversation by asking follow-up questions or start a new query.
