from openai import OpenAI
import streamlit as st

# Tab favicon for the app
st.set_page_config(page_icon='https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png')

# Logo for the side of the app
main_logo ='https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png'
sidebar_logo = 'https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png'
st.logo(main_logo, icon_image=sidebar_logo)

# Title for page
st.title('Chat 💬')

client = OpenAI(api_key=st.secrets["openai_key"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

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