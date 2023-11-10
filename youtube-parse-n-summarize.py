import requests
import spacy
import tensorflow as tf
from transformers import TFBartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

def get_subtitles(video_url):
    """Parses subtitles"""
    try:
        video_id = video_url.split('v=')[1]
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'en'])
        text = ' '.join(transcript['text'] for transcript in transcripts)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_text_to_file(text, output_file):
    """Saves the result to a file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Subtitles text successfully loaded to a file: {output_file}")
    except Exception as e:
        print(f"Error: {e}")

def split_text_into_chunks(text, chunk_size=10000):
    """Splits text into chunks"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_and_add_periods(video_url, output_file):
    """Translates the subtitles to English and adds dots to the text"""
    try:
        video_id = video_url.split('v=')[1]
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'en'])
        subtitles_text = ' '.join(transcript['text'] for transcript in transcripts)

        IAM_TOKEN = 'YOUR_IAM_TOKEN'
        folder_id = 'YOUR_folder_id'
        target_language = 'en'
        chunk_size = 10000

        # Splitting the text into chunks of 10000 characters each
        text_chunks = split_text_into_chunks(subtitles_text, chunk_size)

        translated_texts = []
        for chunk in text_chunks:
            body = {
                "targetLanguageCode": target_language,
                "texts": [chunk],
                "folderId": folder_id,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {0}".format(IAM_TOKEN)
            }

            response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                                     json=body,
                                     headers=headers)

            if response.status_code == 200:
                translated_text = response.json()['translations'][0]['text']
                translated_texts.append(translated_text)
            else:
                print(f"Translation error: {response.text}")

        translated_subtitles = ' '.join(translated_texts)

        # Adding dots
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(translated_subtitles)
            sentences = []
            for sent in doc.sents:
                sentences.append(sent.text.strip())

            text_with_periods = ". ".join(sentences)

            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(text_with_periods)
            print(f"Subtitles with dots saved to a file: {output_file}")

        except Exception as e:
            print(f"Processing error: {e}")

    except Exception as e:
        print(f"Unable to get the subtitles: {e}")

def summarize_text(initial_text, max_length=512, min_length=256, num_beams=10):
    """Summarizes the text"""
    # Checking if CUDA GPU is available and enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Loading pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = TFBartForConditionalGeneration.from_pretrained(model_name)

    # Tokenizing the input text
    inputs = tokenizer.encode("summarize: " + initial_text, return_tensors="tf", max_length=1024, truncation=True)

    # Generating the summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        num_return_sequences=10
    )

    # Decoding the summary tokens back to text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

if __name__ == "__main__":
    video_url = input("Enter a YouTube video link: ")
    output_file = 'output.txt'

    translate_and_add_periods(video_url, output_file)

    with open(output_file, encoding='utf-8') as file:
        initial_text = file.read()

    summarized_text = summarize_text(initial_text)

    with open("summarized_output.txt", 'w', encoding='utf-8') as output:
        output.write(summarized_text)

    print(f"Result with subtitles and summary saved to file: summarized_output.txt")
