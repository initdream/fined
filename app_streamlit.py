import streamlit as st
from chat_pipeline import run_pipeline
import uuid



st.set_page_config(page_title="Chatbot com Haystack", page_icon="🤖")
st.title("Sistema Educacional RAG com Haystack")
st.caption("Conectado ao seu pipeline Haystack e Ollama")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso te ajudar hoje?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Qual o seu prompt?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando documentos..."):
            answer = run_pipeline(prompt, session_id=st.session_state.session_id)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
