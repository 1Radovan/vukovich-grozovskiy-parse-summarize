import tensorflow as tf
from transformers import TFBartForConditionalGeneration, BartTokenizer
# параметры max_length, min_lenght, num_beams и num_return_sequences вляют на длину summary, однако большие значения параметров стоит выставлять от 16gb DDR4+ RAM и мощном процессоре
def summarize_text(initial_text, max_length=4096, min_length=512, num_beams=30):
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
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        num_return_sequences=30
    )

    # Decode the summary tokens back to text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text
# укажите свой путь к файлу с входным текстом
if __name__ == "__main__":
    initial_text = '\n'.join([line for line in
        open('ваш\путь\к\файлу.txt', encoding= 'utf-8')
        ][1:])

    summarized_text = summarize_text(initial_text)
# укажите свой путь к файлу куда будет сохраняться summary
    output = open('output.txt', 'w')

    output.write(summarized_text)

    output.close()
