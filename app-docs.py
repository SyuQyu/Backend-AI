"""
==========================================
API Parafrase Bahasa Indonesia - Dokumentasi
==========================================

Deskripsi:
-----------
API ini melakukan parafrase (pengubahan kalimat dengan makna tetap) untuk teks Bahasa Indonesia menggunakan model HuggingFace T5.

Endpoint:
---------
1. POST /paraphrase
    - Request: { "text": "Teks yang ingin diparafrasekan di sini." }
    - Response: { "result": "Hasil parafrase dari teks Anda." }
    - Maksimal 500 karakter.

2. GET /
    - Response: { "message": "Welcome to the Paraphrasing API. Use /paraphrase to paraphrase text." }

3. GET /health
    - Response: { "status": "ok" }

Cara Menjalankan:
-----------------
1. Install dependensi:
    pip install flask transformers torch sentencepiece

2. Jalankan aplikasi:
    python app-docs.py

3. Akses endpoint:
    - POST http://localhost:8000/paraphrase
    - GET http://localhost:8000/health

Catatan:
--------
- Model: cahya/t5-base-indonesian-summarization-cased
- Output parafrase akan berusaha menjaga panjang dan makna input.
==========================================
"""

from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

MODEL_NAME = "cahya/t5-base-indonesian-summarization-cased"
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
    min_length = int(input_token_len * 0.9)
    max_length = int(input_token_len * 1.1)
    summary_ids = model.generate(
        input_ids,
        min_length=min_length,
        max_length=max_length,
        num_beams=2,
        repetition_penalty=1.5,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        use_cache=True,
        do_sample=True,         # Sampling aktif
        temperature=1.0,        # Sampling lebih acak
        top_k=50,               # Hanya 50 token teratas yang dipertimbangkan
        top_p=0.95,             # Nucleus sampling
        num_return_sequences=3  # Menghasilkan 3 alternatif parafrase
    )

    results = []
    for output in summary_ids:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Hapus awalan jika ada
        for prefix in ["parafrase:", "Parafrase:"]:
            if text.strip().startswith(prefix):
                text = text.strip()[len(prefix):].lstrip()
        results.append(text.strip())

    return jsonify({"result": results})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Paraphrasing API. Use /paraphrase to paraphrase text."})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)