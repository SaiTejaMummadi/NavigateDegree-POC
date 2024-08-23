import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

#App Config
st.set_page_config(page_title = "Navigate")
st.title('Navigate Your Degree. Ask your compass...')

model_choice = st.sidebar.radio("Choose your AI model:", ["OpenAI", "Anthropic", "LLaMA - with Groq", "Mistral"], index=0)


def get_response(user_query, chat_history):
    template = """
    You are a helpful assistant. 
    Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Select the model based on user choice
    if model_choice == "OpenAI":
        llm = ChatOpenAI()
    elif model_choice == "Anthropic":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    elif model_choice == "LLaMA - with Groq":
        llm = ChatGroq(model="llama3-8b-8192")
    elif model_choice == "Mistral":
        llm = ChatMistralAI(model ="mistral-medium")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history" : chat_history,
        "user_question" : user_query,
    })

#session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history= [
        AIMessage(content="Hello, I am a bot. How can I help you? ")
    ]

#conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    else:
        with st.chat_message("Human"):
            st.markdown(message.content)

#user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))