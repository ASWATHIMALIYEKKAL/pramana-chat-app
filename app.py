import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['llmgroq']

def main():

    st.title("Pramana Chat App")

    # Display an image at the top of the app
    st.image(r"pramana-chat-app/image/download.png", 
             caption="Welcome to the Pramana Chat App", use_container_width=True)

    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question:")

    # Initialize chat history in session state if not already done
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if user_question:
        # Process the user question
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        
        # Append to session state chat history
        st.session_state.chat_history.insert(0, message)  # Insert at the beginning for reverse order

    # Display chat history with the latest exchange first
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        st.write("**You:**", message['human'])
        st.write("**Chatbot:**", message['AI'])
        st.write("---")  # Separator for better readability between exchanges

if __name__ == "__main__":
    main()
