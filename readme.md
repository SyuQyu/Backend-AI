# API Parafrase Bahasa Indonesia

## Deskripsi

API ini melakukan parafrase (pengubahan kalimat dengan makna tetap) untuk teks Bahasa Indonesia menggunakan model HuggingFace T5 (`cahya/t5-base-indonesian-summarization-cased`).

---

## Endpoint

### 1. POST `/paraphrase`

- **Request:**
  ```json
  { "text": "Teks yang ingin diparafrasekan di sini." }
  ```
  - Maksimal 500 karakter.
- **Response:**
  ```json
  { "result": "Hasil parafrase dari teks Anda." }
  ```

### 2. GET `/`

- **Response:**
  ```json
  {
    "message": "Welcome to the Paraphrasing API. Use /paraphrase to paraphrase text."
  }
  ```

### 3. GET `/health`

- **Response:**
  ```json
  { "status": "ok" }
  ```

---

## Cara Menjalankan

1. **Install dependensi:**

   ```
   pip install flask transformers torch sentencepiece
   ```

2. **Jalankan aplikasi:**

   ```
   python app-docs.py
   ```

3. **Akses endpoint:**
   - Paraphrase: `POST http://localhost:8000/paraphrase`
   - Health check: `GET http://localhost:8000/health`

---

## Penjelasan Model

- **Model:** `cahya/t5-base-indonesian-summarization-cased`
- **Tokenizer:** `T5Tokenizer` dari HuggingFace, mengubah teks ke token.
- **Model:** `T5ForConditionalGeneration`, tipe encoder-decoder untuk sequence-to-sequence.
- **Prompt:** Input diubah menjadi `"parafrase: <teks>"` agar model memahami tugas parafrase.
- **Output:** Model menghasilkan teks baru yang maknanya tetap, namun dengan susunan kata berbeda.

---

## Penjelasan Parameter `model.generate`

| Parameter                 | Penjelasan                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| `input_ids`               | Token hasil encoding dari prompt input.                                                          |
| `min_length`              | Jumlah token minimum pada output.                                                                |
| `max_length`              | Jumlah token maksimum pada output.                                                               |
| `num_beams`               | Jumlah beam pada beam search (2 = kompromi kualitas & kecepatan).                                |
| `repetition_penalty`      | Penalti untuk pengulangan kata/frasa pada hasil output (>1 = lebih ketat).                       |
| `length_penalty`          | Preferensi output panjang/pendek (1.0 = netral).                                                 |
| `no_repeat_ngram_size`    | Melarang pengulangan n-gram sepanjang nilai ini (misal 2 = bigram).                              |
| `use_cache`               | Mengaktifkan cache untuk mempercepat decoding.                                                   |
| `do_sample`               | Jika `False`, hasil lebih stabil dan cepat.                                                      |
| `temperature`             | Mengatur randomness saat sampling (`do_sample=True`). Semakin tinggi, semakin acak. Default: 1.0 |
| `top_k`                   | Saat sampling, hanya pertimbangkan `k` token dengan probabilitas tertinggi. Default: 50          |
| `top_p`                   | Saat sampling, hanya pertimbangkan token dengan probabilitas kumulatif hingga `p`. Default: 1.0  |
| `early_stopping`          | Jika `True`, beam search akan berhenti saat semua beam mencapai token akhir.                     |
| `bad_words_ids`           | Daftar token yang tidak boleh muncul di output.                                                  |
| `num_return_sequences`    | Jumlah output berbeda yang dihasilkan untuk setiap input.                                        |
| `decoder_start_token_id`  | Token awal untuk decoder (penting untuk beberapa model).                                         |
| `bos_token_id`            | Token awal (beginning of sentence).                                                              |
| `eos_token_id`            | Token akhir (end of sentence).                                                                   |
| `pad_token_id`            | Token padding.                                                                                   |
| `output_scores`           | Jika `True`, mengembalikan skor probabilitas output.                                             |
| `return_dict_in_generate` | Jika `True`, mengembalikan output dalam bentuk dictionary.                                       |
| `forced_bos_token_id`     | Memaksa token awal tertentu di output.                                                           |
| `forced_eos_token_id`     | Memaksa token akhir tertentu di output.                                                          |

---

## Contoh Kode Backend

```python
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
        do_sample=False
    )
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"result": summary_text.strip()})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Paraphrasing API. Use /paraphrase to paraphrase text."})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

---

## Catatan

- Maksimal input adalah 500 karakter.
- Output parafrase akan berusaha menjaga panjang dan makna input, namun hasil sangat tergantung pada model.
- Untuk performa lebih baik, gunakan GPU jika tersedia.

---

## Contoh Request

```bash
curl -X POST http://localhost:8000/paraphrase \
     -H "Content-Type: application/json" \
     -d '{"text": "Saya suka makan nasi goreng di pagi hari."}'
```

**Response:**

```json
{
  "result": "Di pagi hari, saya senang menyantap nasi goreng."
}
```
