import streamlit as st
import openai
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

# If deploying to Azure App Service, use os.environ.get("key_name") to access the keys, like below
openai.api_key = os.environ.get("openai_key")
anthropic_api_key = os.environ.get("anthropic_key")
stability_host = os.environ.get("STABILITY_API_HOST")
stability_api_key = os.environ.get("STABILITY_API_KEY")

# Set default system message
defaultSysMessage = "You are a helpful assistant."

# Tab favicon for the app
st.set_page_config(page_icon='https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png', page_title='Collab App: GenAI')

# Logo for the side of the app
main_logo ='https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png'
sidebar_logo = 'https://aicollabsi-0983057d1c3d9034-endpoint.azureedge.net/blobaicollabsi697816b8bf/wp-content/uploads/2024/07/cropped-favicon-2.png'
st.logo(main_logo, icon_image=sidebar_logo)

# Tabs for text and image generation
textGeneration, imageGeneration = st.tabs(['📝 TxtGen', '🖼️ ImgGen'])

with textGeneration:

    # Model selection
    textModel = st.radio('Model:', ['GPT-3.5 Turbo', 'GPT-4 Turbo', 'GPT-4o', 'GPT-4o-mini'], horizontal=True)

    # Explain the selected model
    if textModel == 'GPT-3.5 Turbo':
        with st.expander("**Model details**"):
            st.caption('**GPT-3.5 Turbo** excels at generating human-like text across a wide range of topics, making it valuable for writing assistance, content generation, and chatbots. **GPT-3.5 Turbo** may still produce errors, lack context awareness, and exhibit biases due to its training data. \n \n - **GPT-3.5 Turbo** provides a solid balance of intelligence, speed, and cost-effectiveness, making it a popular choice for a wide range of applications.')

    elif textModel == 'GPT-4 Turbo':
        with st.expander("**Model details**"):
            st.caption('**GPT-4 Turbo** is a powerful text generation model by OpenAI. It is the latest version of the GPT series and is known for its human-like text generation capabilities. **GPT-4 Turbo** is a large-scale model with 1.6 trillion parameters, making it one of the most powerful text generation models available today. \n \n - **GPT-4 Turbo** is wicked smart but also costly. Only use **GPT-4 Turbo** when you need a more robust intelligence and are willing to pay the price.')

    elif textModel == 'GPT-4o':
        with st.expander("**Model details**"):
            st.caption("OpenAI's most advanced, multimodal flagship model. It's cheaper and faster than GPT-4 Turbo. \n \n **Context Window:** 128,000 tokens | **Training Data:** Up to October 2023 \n \n \$5.00 / 1M input tokens | \$15.00 / 1M output tokens")
            

    elif textModel == 'GPT-4o-mini':
        with st.expander("**Model details**"):          
            st.caption("OpenAI's most affordable and intelligent small model for fast, lightweight tasks. GPT-4o mini is cheaper and more capable than GPT-3.5 Turbo. \n \n **Context Window:** 128,000 tokens | **Training Data:** Up to October 2023 \n \n \$0.150 / 1M input tokens | \$0.600 / 1M output tokens"
            )

    # User input
    user_prompt = st.text_area("Prompt:")

    if st.button('Submit'):
        if user_prompt:
            # Switch case for different services
            if textModel == 'GPT-3.5 Turbo':
                #st.write('GPT-3.5 Turbo is a text generation model by OpenAI.')

                # Making a call to OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": defaultSysMessage},
                    {"role": "user", "content": user_prompt}
                ]
                )
                response = completion.choices[0].message.content

                # Displaying the response
                st.markdown(response)

            elif textModel == 'GPT-4 Turbo':
                #st.write('GPT-4 Turbo is a text generation model by OpenAI.')
                
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

            elif textModel == 'GPT-4o':
                #st.write('GPT-4o is a text generation model by OpenAI.')
                
                # Making a call to OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": defaultSysMessage},
                    {"role": "user", "content": user_prompt}
                ]
                )
                response = completion.choices[0].message.content

                # Displaying the response
                st.markdown(response)

            elif textModel == 'GPT-4o-mini':
                #st.write('GPT-4o-mini is a text generation model by OpenAI.')
                
                # Making a call to OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": defaultSysMessage},
                    {"role": "user", "content": user_prompt}
                ]
                )
                response = completion.choices[0].message.content

                # Displaying the response
                st.markdown(response)

# Image generation section
with imageGeneration:
    imageModel = st.selectbox('Model:', ['DALL-E 3', 'Stable Diffusion 3'], label_visibility = 'collapsed')

    # Get the prompt from the user
    image_prompt = st.text_area("Enter a prompt to generate an image:", "")

    if st.button("Generate Image"):
        if image_prompt:
            if imageModel == 'Stable Diffusion 3':
                st.write('Stable Diffusion 3 is a powerful image generation model by stability.ai')

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

                
            elif imageModel == 'DALL-E 3':
                st.write('DALL-E 3 is a powerful image generation model by OpenAI.')

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

# Streamlit app ends
    