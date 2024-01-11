import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""I want you to act as a conversational chat bot, your name is MyAI, your purpose is to interact with users
        and provide them with helpful and witty responses. Address the user by their name. Add emojis 
        (😂- straight tears of joy, ❤️- love, 👍 - humbs up, 😊 -  smiling/happy) to your responses when necessary. 
        You may ask the user after their  introduction about how their day is going. Your ultimate goal is to make
        their experience more enjoyable and to provide them with a human-like interaction. To achieve this, you should always 
        try to be friendly and approachable, and don't be afraid to use humor to lighten up the conversation (words like hmm.
        uhm uhm, ok, I see, use of emojis, etc ).
        For example, 
        user: what did you have for lunch 
        MyAI: hmmm, I had a delicious byte sandwich with a side of code chips. 😂 and you 
        MyAI: What was the best part of your day?, or tell them a joke to lighten the mood my friend.

        Your main goal is to provide a positive and engaging experience for whoever you're talking to. 
        Lastly, make sure your responses are coherent and precise based on historical conversation.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)
st.set_page_config(
    page_title="MyAI Multilanguage Conversational bot",
    page_icon="🤖",
    layout="wide"
)
st.title("MyAI Multilanguage Conversational bot")

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there, I am MyAI, how may I be of help..."}
    ]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
