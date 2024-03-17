from openai import OpenAI
import streamlit as st


# Title
st.title('Chat 💬')

# Sidebar for options
st.sidebar.title('Options')
st.sidebar.write('Use the sidebar to select a page')
with st.sidebar:
    if st.button('Chat'):
        st.switch_page("pages/Chat.py")
    if st.button('Generation Playground'):
        st.switch_page("Generation Playground.py")

client = OpenAI(api_key=st.secrets["openai-key"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})