import streamlit as st
import openai
import anthropic
import numpy as np
import io
from PIL import Image
import base64
import requests
import os

# Set API service urls and keys from secrets.toml file in CWD
# If testing locally, use st.secrets["key_name"] to access the keys, like below
#openai.api_key = st.secrets["openai_key"]
#anthropic_api_key = st.secrets["anthropic_key"]
#stability_host = st.secrets["STABILITY_API_HOST"]
#stability_api_key = st.secrets["STABILITY_API_KEY"]

# If deploying to Azure App Service, use os.environ.get("key_name") to access the keys
openai.api_key = os.environ.get("openai_key")
anthropic_api_key = os.environ.get("anthropic_key")
stability_host = os.environ.get("STABILITY_API_HOST")
stability_api_key = os.environ.get("STABILITY_API_KEY")

# Set default system message
defaultSysMessage = "You are a helpful assistant."

# Streamlit app starts
st.title('Generation Playground')

# Heading for the app
st.write('This is a simple web app to play with different AI models.')

# Sidebar for options
st.sidebar.title('Options')
st.sidebar.write('Use the sidebar to select a page')
with st.sidebar:
    if st.button('Chat'):
        st.switch_page("pages/Chat.py")
    if st.button('Generation Playground'):
        st.switch_page("Generation Playground.py")

# Tabs for text and image generation
textGeneration, imageGeneration, chat = st.tabs(['📝 Text Generation', '🖼️ Image Generation', '💬 Chat'])

with textGeneration:

    # Model selection
    st.subheader('**Model selection**')
    textModel = st.selectbox('Text Model:', ['GPT-3.5 Turbo', 'GPT-4 Turbo', 'Claude 3 Haiku'], label_visibility = 'collapsed')

    # Explain the selected model
    if textModel == 'GPT-3.5 Turbo':
        with st.expander("**Model explanation**", expanded=True):
            st.caption('**GPT-3.5 Turbo** excels at generating human-like text across a wide range of topics, making it valuable for writing assistance, content generation, and chatbots. **GPT-3.5 Turbo** may still produce errors, lack context awareness, and exhibit biases due to its training data. \n \n - **GPT-3.5 Turbo** provides a solid balance of intelligence, speed, and cost-effectiveness, making it a popular choice for a wide range of applications.')

    elif textModel == 'GPT-4 Turbo':
        with st.expander("**Model Explanation**", expanded=True):
            st.caption('**GPT-4 Turbo** is a powerful text generation model by OpenAI. It is the latest version of the GPT series and is known for its human-like text generation capabilities. **GPT-4 Turbo** is a large-scale model with 1.6 trillion parameters, making it one of the most powerful text generation models available today. \n \n - **GPT-4 Turbo** is wicked smart but also costly. Only use **GPT-4 Turbo** when you need a more robust intelligence and are willing to pay the price.')

    elif textModel == 'Claude 3 Haiku':
        with st.expander("**Model explanation**", expanded=True):
            st.caption("**Claude 3 Haiku** is a text generation model by Anthropic. It is designed to generate haikus, a form of Japanese poetry. The model is trained on a large dataset of haikus and is capable of generating high-quality haikus with a human-like touch. \n \n - **Claude 3 Haiku** is a specialized model that is perfect for generating haikus. It is not as versatile as GPT-3.5 Turbo or GPT-4 Turbo, but it excels at generating haikus with a human-like touch.")

    # User input
    user_prompt = st.text_area("Enter your prompt:")

    if st.button('Submit'):
        if user_prompt:
            # Switch case for different services
            if textModel == 'GPT-3.5 Turbo':
                st.write('GPT-3.5 Turbo is a text generation model by OpenAI.')

                # Making a call to OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": defaultSysMessage},
                    {"role": "user", "content": user_prompt}
                ]
                )
                # Displaying the response
                st.text_area("GPT-3.5 Response:", value=completion.choices[0].message.content, height=250)

            elif textModel == 'GPT-4 Turbo':
                st.write('GPT-4 Turbo is a text generation model by OpenAI.')
                
                # Making a call to OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": defaultSysMessage},
                    {"role": "user", "content": user_prompt}
                ]
                )
                response = completion.choices[0].message.content
                # Displaying the response
                st.markdown(response)

            elif textModel == 'Claude 3 Haiku':
                st.write('Claude 3 Haiku is a text generation model by Anthropic.')

                # Making a call to Anthropic API
                client = anthropic.Anthropic(

                api_key=anthropic_api_key,
                )
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0,
                    system=defaultSysMessage,
                    messages=[{ "role": "user", "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }]
                )
                st.text_area("Claude 3 Haiku Response:", value=message.content[0].text, height=250)
            
        else:
            st.warning('Please enter a prompt.')

# Image generation section
with imageGeneration:
    imageModel = st.selectbox('Model:', ['OpenAI DALL-E 3', 'Stability AI'], label_visibility = 'collapsed')

    # Get the prompt from the user
    image_prompt = st.text_area("Enter a prompt to generate an image:", "")

    if st.button("Generate Image"):
        if image_prompt:
            if imageModel == 'Stability AI':
                st.write('Stability AI is a powerful image generation model by OpenAI.')

                # Making a call to Stability AI API
                engine_id = "stable-diffusion-v1-6"
                api_host = stability_host
                api_key = stability_api_key

                if api_key is None:
                    raise Exception("Missing Stability API key.")

                response = requests.post(
                    f"{api_host}/v1/generation/{engine_id}/text-to-image",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json={
                        "text_prompts": [
                            {
                                "text": image_prompt
                            }
                        ],
                        "cfg_scale": 7,
                        "height": 1024,
                        "width": 1024,
                        "samples": 1,
                        "steps": 30,
                    },
                )

                if response.status_code != 200:
                    raise Exception("Non-200 response: " + str(response.text))

                data = response.json()

                for i, image in enumerate(data["artifacts"]):
                    img_data = base64.b64decode(image["base64"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, caption=image_prompt, use_column_width=True)
                    btn = st.download_button(
                            label="Download image",
                            data=img_data,
                            file_name=f"{image_prompt}.png",
                            mime="image/png"
                        )

                
            elif imageModel == 'OpenAI DALL-E 3':
                st.write('OpenAI DALL-E 3 is a powerful image generation model by OpenAI.')

                # Making a call to OpenAI API
                imageClient = openai.OpenAI(api_key=openai.api_key)
                
                # Call the DALL-E 3 API to generate the image
                
                response = imageClient.images.generate(
                    model="dall-e-3",
                    prompt=image_prompt,
                    n=1,
                    size="1024x1024",
                    response_format="b64_json"
                )

                # Get the generated image URL
                image_url = response.data[0].b64_json

                # Decode the base64 image data and display it
                image_data = io.BytesIO(base64.b64decode(image_url))
                image = Image.open(image_data)
                st.image(image, caption=image_prompt, use_column_width=True)
                btn = st.download_button(
                        label="Download image",
                        data=image_data,
                        file_name=f"{image_prompt}.png",
                        mime="image/png"
                    )

        else:
            st.warning('Please enter a prompt.')

with chat:
    #chatModel = st.selectbox('Chat Model:', ['GPT-3.5 Turbo', 'GPT-4 Turbo $$', 'Claude 3 Haiku'])

    chatClient = openai.OpenAI(api_key=openai.api_key)

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
            stream = chatClient.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Streamlit app ends
    