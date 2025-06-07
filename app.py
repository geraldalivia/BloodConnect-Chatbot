import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import os
import sys # Import sys untuk print ke stderr

# Base Directory untuk folder final_model
MODEL_PATH = "./final_model"

# Load Resources
def load_chatbot_resources():
    try:
        print("Memulai pemuatan sumber daya chatbot...")
        print(f"Memeriksa keberadaan folder model di: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Folder model tidak ditemukan di {MODEL_PATH}.", file=sys.stderr)
            raise FileNotFoundError(f"Folder model tidak ditemukan di {MODEL_PATH}")

        print("Memuat model dan tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        # Pindahkan model ke GPU jika tersedia di Spaces
        if torch.cuda.is_available():
            model.to('cuda')
            print("Model berhasil dimuat ke GPU.")
        else:
            print("GPU tidak tersedia, model dimuat ke CPU.")

        print("Model dan tokenizer berhasil dimuat.")

        # Memuat Dataset FAQ dari CSV
        faq_file_path = "dataset/bloodconnect_faq_valid.csv" 
        print(f"Memeriksa keberadaan file FAQ di: {faq_file_path}")

        if not os.path.exists(faq_file_path):
            print(f"Error: File FAQ tidak ditemukan di {faq_file_path}.", file=sys.stderr)
            raise FileNotFoundError(f"File FAQ tidak ditemukan di {faq_file_path}. Pastikan sudah diunggah.")
        else:
            print(f"Memuat dataset FAQ dari: {faq_file_path}...")
            df_faq = pd.read_csv(faq_file_path)
            print("Dataset FAQ berhasil dimuat.")

        print("Memuat model SentenceTransformer untuk embedding teks...")
        
        # Model embedding akan start
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # Pastikan model ini dapat diakses/diunduh
        print("Model SentenceTransformer berhasil dimuat.")

        print("Membuat embedding untuk pertanyaan FAQ...")
        faq_questions = df_faq['short_question'].tolist()
        faq_question_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)
        print("Embedding pertanyaan FAQ selesai.")

        print("Pemuatan sumber daya chatbot selesai.")
        return tokenizer, model, embedding_model, df_faq, faq_questions, faq_question_embeddings

    except Exception as e:
        print(f"*** ERROR KRITIS SAat MEMUAT SUMBER DAYA: {e} ***", file=sys.stderr)
        print("Pastikan semua file model dan FAQ berada di jalur yang benar.", file=sys.stderr)
        print("Periksa juga ketersediaan memori dan koneksi internet di Hugging Face Spaces.", file=sys.stderr)
        
        # Feedback exception
        raise RuntimeError(f"Gagal memuat model atau sumber daya chatbot: {e}") from e


# Memanggil fungsi pemuatan sumber daya
try:
    tokenizer_global, model_global, embedding_model_global, df_faq_global, faq_questions_global, faq_question_embeddings_global = load_chatbot_resources()
except RuntimeError as e:
    
    # Tangani error startup agar antarmuka Gradio bisa tampil dengan pesan error
    gr.Warning(str(e))
    
    # Sebagai fallback
    tokenizer_global, model_global, embedding_model_global, df_faq_global, faq_questions_global, faq_question_embeddings_global = None, None, None, None, None, None


# Fungsi Pencarian Relevansi FAQ (Retrieval)
def find_relevant_faq(user_question, faq_questions, faq_question_embeddings, threshold=0.7):
    if embedding_model_global is None: # Pastikan model embedding dimuat
        print("find_relevant_faq: embedding_model_global is None", file=sys.stderr)
        return None
    user_question_embedding = embedding_model_global.encode(user_question, convert_to_tensor=True)
    similarities = cosine_similarity(user_question_embedding.unsqueeze(0).cpu(), faq_question_embeddings.cpu())[0]
    most_similar_idx = similarities.argmax()
    max_similarity = similarities[most_similar_idx]

    if max_similarity >= threshold:
        return df_faq_global.loc[most_similar_idx, 'short_answer']
    else:
        return None


# Fungsi Generate Jawaban dengan Model CLM
def generate_answer_with_clm(prompt, model, tokenizer, max_length=100):
    if model_global is None or tokenizer_global is None: # Pastikan model CLM dimuat
        print("generate_answer_with_clm: model_global or tokenizer_global is None", file=sys.stderr)
        return "Model generasi tidak dapat dimuat."
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    output_sequences = model.generate(
        input_ids=inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text

    
# Fungsi Utama
def chatbot_interaction(user_input, chat_history):
    # Ini akan mencegah error jika sumber daya tidak berhasil dimuat saat startup
    if model_global is None or tokenizer_global is None or embedding_model_global is None:
        return chat_history + [(user_input, "Maaf, sistem chatbot sedang tidak dapat diakses. Silakan coba lagi nanti. (Error: Resources not loaded)")], ""

    faq_answer = find_relevant_faq(user_input, faq_questions_global, faq_question_embeddings_global)

    if faq_answer:
        response = faq_answer
    else:
        # Menampilkan pesan di UI jika tidak ada jawaban FAQ yang relevan
        # Ini akan muncul di konsol Gradio, bukan di chat bubble secara langsung
        print("Tidak ada jawaban FAQ yang relevan. Mencoba menghasilkan jawaban dengan model...")
        prompt = f"Pertanyaan: {user_input}\nJawaban:"
        generated_answer = generate_answer_with_clm(prompt, model_global, tokenizer_global)
        if generated_answer:
            response = f"Maaf, saya tidak menemukan jawaban persis di FAQ. Mungkin ini bisa membantu: {generated_answer}"
        else:
            response = "Maaf, saya tidak dapat menemukan atau menghasilkan jawaban untuk pertanyaan Anda saat ini."

    chat_history.append((user_input, response))
    return chat_history, "" # Return chat_history and empty string for textbox


# Antarmuka Gradio
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container { max-width: 700px; margin: auto; padding: 20px; }") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #DC2626;">🩸 Asisten Donor Darah BloodConnect</h1>
        <p style="text-align: center; color: #4A5568;">Tanyakan apa pun tentang aplikasi BloodConnect atau informasi seputar donor darah!</p>
        """
    )

    # Komponen Chatbot Gradio
    chatbot = gr.Chatbot(
        label="Chatbot BloodConnect",
        height=400, 
        show_copy_button=True,
        bubble_full_width=False 
    )

    # Input teks untuk pengguna
    msg = gr.Textbox(
        label="Tulis pertanyaan Anda di sini...",
        placeholder="Contoh: Apa itu aplikasi BloodConnect? atau Apa saja syarat donor darah?",
        container=False 
    )

    with gr.Row():
        send_btn = gr.Button("Kirim Pertanyaan", variant="primary") 
        clear_btn = gr.ClearButton([msg, chatbot], value="Bersihkan Chat")

    # Fungsi untuk menangani input pengguna dan memperbarui chat
    send_btn.click(chatbot_interaction, [msg, chatbot], [chatbot, msg])
    msg.submit(chatbot_interaction, [msg, chatbot], [chatbot, msg]) 


# Jalankan aplikasi Gradio
if __name__ == "__main__":
    demo.launch()