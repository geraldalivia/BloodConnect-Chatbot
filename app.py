import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import os
import sys

# Konfigurasi Streamlit untuk Header Halaman
st.set_page_config(
    page_title="BloodConnect Chatbot",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load Dataset
@st.cache_resource
def load_chatbot_resources():
    try:
        MODEL_PATH = "./final_model"
        FAQ_FILE_PATH = "dataset/bloodconnect_faq_valid.csv"

        st.spinner("Memulai pemuatan sumber daya chatbot...")
        
        # Cek keberadaan folder model
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Folder model tidak ditemukan di {MODEL_PATH}.")
            raise FileNotFoundError(f"Folder model tidak ditemukan di {MODEL_PATH}")

        # Memuat model dan tokenizer
        st.write("Memuat model dan tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        # Pindahkan model ke GPU jika tersedia
        if torch.cuda.is_available():
            model.to('cuda')
            st.write("Model berhasil dimuat ke GPU.")
        else:
            st.write("GPU tidak tersedia, model dimuat ke CPU.")

        st.write("Model dan tokenizer berhasil dimuat.")

        # Memuat Dataset FAQ dari CSV
        if not os.path.exists(FAQ_FILE_PATH):
            st.error(f"Error: File FAQ tidak ditemukan di {FAQ_FILE_PATH}.")
            raise FileNotFoundError(f"File FAQ tidak ditemukan di {FAQ_FILE_PATH}. Pastikan sudah diunggah.")
        else:
            st.write(f"Memuat dataset FAQ dari: {FAQ_FILE_PATH}...")
            df_faq = pd.read_csv(FAQ_FILE_PATH)
            st.write("Dataset FAQ berhasil dimuat.")

        st.write("Memuat model SentenceTransformer untuk embedding teks...")
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
        st.write("Model SentenceTransformer berhasil dimuat.")

        st.write("Membuat embedding untuk pertanyaan FAQ...")
        faq_questions = df_faq['short_question'].tolist()
        faq_question_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)
        st.write("Embedding pertanyaan FAQ selesai.")

        st.success("Pemuatan sumber daya chatbot selesai!")
        return tokenizer, model, embedding_model, df_faq, faq_questions, faq_question_embeddings

    except Exception as e:
        st.error(f"*** ERROR KRITIS SAAT MEMUAT SUMBER DAYA: {e} ***")
        st.write("Pastikan semua file model dan FAQ berada di jalur yang benar.")
        st.write("Periksa juga ketersediaan memori dan koneksi internet di Hugging Face Spaces.")
        st.exception(e) # Menampilkan traceback
        st.stop() # Menghentikan eksekusi Streamlit jika ada error fatal


# Memanggil fungsi load dataset
if "resources_loaded" not in st.session_state:
    st.session_state.tokenizer, \
    st.session_state.model, \
    st.session_state.embedding_model, \
    st.session_state.df_faq, \
    st.session_state.faq_questions, \
    st.session_state.faq_question_embeddings = load_chatbot_resources()
    st.session_state.resources_loaded = True


# Mengakses Dataset
tokenizer_global = st.session_state.tokenizer
model_global = st.session_state.model
embedding_model_global = st.session_state.embedding_model
df_faq_global = st.session_state.df_faq
faq_questions_global = st.session_state.faq_questions
faq_question_embeddings_global = st.session_state.faq_question_embeddings


# Fungsi untuk Generate jawaban chatbot
def find_relevant_faq(user_question, faq_questions, faq_question_embeddings, threshold=0.7):
    if embedding_model_global is None:
        return None
    user_question_embedding = embedding_model_global.encode(user_question, convert_to_tensor=True)
    similarities = cosine_similarity(user_question_embedding.unsqueeze(0).cpu(), faq_question_embeddings.cpu())[0]
    most_similar_idx = similarities.argmax()
    max_similarity = similarities[most_similar_idx]

    if max_similarity >= threshold:
        return df_faq_global.loc[most_similar_idx, 'short_answer']
    else:
        return None

def generate_answer_with_clm(prompt, model, tokenizer, max_length=100):
    if model_global is None or tokenizer_global is None:
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

# UI untuk streamlit
st.markdown(
    """
    <style>
    .reportview-container {
        flex-direction: column;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .st-emotion-cache-1pxazr7 { /* Adjust padding of the main content area */
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-vk337c { /* Padding above the chat messages */
        padding-bottom: 0px !important;
    }
    .st-emotion-cache-l4e44q { /* Margin for the whole chat input container */
        margin-bottom: 15px !important;
    }
    /* Styles for chat bubbles */
    .chat-bubble {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 80%;
        position: relative;
        font-size: 0.95em;
        line-height: 1.4;
    }
    .user-bubble {
        background-color: #262626; /* Darker grey for user */
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    .bot-bubble {
        background-color: #0c0d0d; /* Light grey for bot */
        color: #fff;
        align-self: flex-start;
        margin-right: auto;
        border-bottom-left-radius: 2px;
    }
    .st-emotion-cache-16yaunp { /* target specific Streamlit markdown element that wraps message */
        padding: 0px !important;
        margin: 0px !important;
    }
    .st-emotion-cache-gq8p36 { /* Target columns to remove padding */
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩸 Asisten Donor Darah BloodConnect")
st.markdown("<p style='text-align: center; color: #ADB5BD;'>Tanyakan apa pun tentang aplikasi BloodConnect atau informasi seputar donor darah!</p>", unsafe_allow_html=True)


# Inisialisasi riwayat obrolan di session_state jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Pesan pembuka dari AI
    st.session_state.messages.append({"role": "assistant", "content": "Halo! Saya adalah Asisten Donor Darah. Ada yang bisa saya bantu terkait donor darah?"})


# Tampilkan pesan obrolan sebelumnya
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="chat-bubble bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)

# Input pengguna
user_input = st.chat_input("Tulis pertanyaan Anda di sini...")

if user_input:
    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)

    # Dapatkan respons dari chatbot
    with st.chat_message("assistant"):
        with st.spinner("Sedang memproses..."):
            if model_global is None or tokenizer_global is None or embedding_model_global is None:
                response = "Maaf, sistem chatbot sedang tidak dapat diakses. Silakan coba lagi nanti. (Error: Resources not loaded)"
            else:
                faq_answer = find_relevant_faq(user_input, faq_questions_global, faq_question_embeddings_global)

                if faq_answer:
                    response = faq_answer
                else:
                    prompt = f"Pertanyaan: {user_input}\nJawaban:"
                    generated_answer = generate_answer_with_clm(prompt, model_global, tokenizer_global)
                    if generated_answer and generated_answer != "Model generasi tidak dapat dimuat.":
                        response = f"Maaf, saya tidak menemukan jawaban persis di FAQ. Mungkin ini bisa membantu: {generated_answer}"
                    else:
                        response = "Maaf, saya tidak dapat menemukan atau menghasilkan jawaban untuk pertanyaan Anda saat ini."
            
            st.markdown(f'<div class="chat-bubble bot-bubble">{response}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})
