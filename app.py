from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# MODEL_NAME = "indonlp/cendol-mt5-small-chat"
# MODEL_NAME = "Wikidepia/IndoT5-base-paraphrase"
MODEL_NAME = "cahya/t5-base-indonesian-summarization-cased"
# MODEL_NAME = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    data = request.get_json()
    text = data.get("text", "")
    if len(text) > 500:
        return jsonify({"error": "Teks maksimal 500 karakter."}), 400
    prompt = f"parafrase: {text}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_char_len = len(text)

    # Estimasi token ke karakter (umumnya 1 token ≈ 3-4 karakter untuk bahasa Indonesia)
    # Kita targetkan jumlah token output ≈ jumlah token input
    input_token_len = input_ids.shape[1]
    # min_length = input_token_len
    # max_length = input_token_len + 10  # beri toleransi sedikit
    min_length = int(input_token_len * 0.9)
    max_length = int(input_token_len * 1.1)
    summary_ids = model.generate(
        input_ids,
        min_length=min_length,
        max_length=max_length,
        num_beams=2,              # Lebih cepat
        repetition_penalty=1.5,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        use_cache=True,
        do_sample=False
        # # input_ids,
        # # min_length=20,
        # # max_length=128,
        # num_beams=10,
        # repetition_penalty=2.5,
        # length_penalty=1.0,
        # early_stopping=True,
        # no_repeat_ngram_size=2,
        # use_cache=True,
        # do_sample=True,
        # temperature=0.8,
        # top_k=50,
        # top_p=0.95
        # input_ids,
        # min_length=40,          # hasil lebih panjang
        # max_length=500,         # hasil lebih panjang
        # num_beams=3,            # lebih cepat dari 10
        # repetition_penalty=2.0,
        # length_penalty=1.0,
        # early_stopping=True,
        # no_repeat_ngram_size=2,
        # use_cache=True,
        # do_sample=False  
    )
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Jika hasil terlalu jauh dari panjang input, bisa diulang (opsional)
    # while abs(len(summary_text) - input_char_len) > 50:
    #     summary_ids = model.generate(
    #         input_ids,
    #         min_length=min_length,
    #         max_length=max_length+10,
    #         ...
    #     )
    #     summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({"result": summary_text.strip()})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Paraphrasing API. Use /paraphrase to paraphrase text."})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)