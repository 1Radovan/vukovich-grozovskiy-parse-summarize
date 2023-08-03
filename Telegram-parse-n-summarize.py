import telethon
import asyncio
from datetime import datetime, timedelta, timezone
import requests
import tensorflow as tf
from transformers import TFBartForConditionalGeneration, BartTokenizer

IAM_TOKEN = 't1.9euelZqMk5CYzc_KmJDLyY7Ljcadx-3rnpWazJ3Hk5WLj5iKysyQlM2Ni4vl8_djA1JZ-e82XCkv_d3z9yMyT1n57zZcKS_9zef1656Vmsqdi46NjpOXls7Hx4zKnpPO7_zF656Vmsqdi46NjpOXls7Hx4zKnpPO.4w_hjBZZEQMDK3YwGcuxe3Bw2z_XMiFMBC6sOsU_LQbRBSJIjMvz_bQ2MFyVsvsOzVc_wu2Kz5kx5EJfBm8QBA'
folder_id = 'b1grp573mgkcni8josbq'
target_language = 'en'
texts = []

async def translate_text(text):
    body = {
        "targetLanguageCode": target_language,
        "texts": [text],
        "folderId": folder_id,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {0}".format(IAM_TOKEN)
    }

    try:
        response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                                 json=body,
                                 headers=headers)

        if response.status_code == 200:
            translated_text = response.json()["translations"][0]["text"]
            return translated_text
        else:
            print("Translation failed: ", response.text)
            return None
    except requests.RequestException as e:
        print("Translation error: ", e)
        return None

async def summarize_text(initial_text):
    # Check if CUDA GPU is available and enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Load pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = TFBartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + initial_text, return_tensors="tf", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=512,
        min_length=256,
        num_beams=10,
        early_stopping=True,
        num_return_sequences=10
    )

    # Decode the summary tokens back to text
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summarized_text

async def main():
    # Telegram API credentials
    api_id = 27582586  # Replace with your API ID
    api_hash = '8f582dcc191f7d6c89db6c93a7776059'  # Replace with your API Hash

    # Telegram channel username or chat ID
    channel_name = 'your_channel_name'  # Replace with the channel username or chat ID

    client = telethon.TelegramClient('session_name', api_id, api_hash)

    try:
        await client.start()

        if not client.is_user_authorized():
            print("Client not authorized. Please check your API credentials.")
            return

        if not client.is_connected():
            print("Client not connected. Attempting to connect...")
            await client.connect()

        print("Client connected and authorized.")

        channel = await client.get_entity(channel_name)

        # Specify the start and end date of the message period you want to scrape.
        # For example, from 1st July to 31st July 2023.
        start_date = datetime(2023, 7, 1).replace(tzinfo=timezone.utc)
        end_date = datetime(2023, 7, 31).replace(tzinfo=timezone.utc)

        with open('Scraped.txt', 'a', encoding='utf-8') as txt_file:
            async for message in client.iter_messages(channel):
                if start_date <= message.date <= end_date:
                    message_text = message.message
                    # Split the message into smaller parts for translation
                    for i in range(0, len(message_text), 10000):
                        chunk = message_text[i:i + 10000]
                        translated_text = await translate_text(chunk)
                        if translated_text:
                            txt_file.write(translated_text + '\n')

        with open('Scraped.txt', 'r', encoding='utf-8') as input_file:
            initial_text = input_file.read()

        # Generate the summary
        summarized_text = await summarize_text(initial_text)

        with open('output.txt', 'w', encoding='utf-8') as output_file:
            output_file.write(summarized_text)

    except telethon.errors.rpcerrorlist.AuthKeyUnregisteredError:
        print("AuthKeyUnregisteredError: The key is not registered in the system.")
    except Exception as e:
        print("An error occurred: ", e)
    finally:
        await client.log_out()

if __name__ == '__main__':
    asyncio.run(main())
