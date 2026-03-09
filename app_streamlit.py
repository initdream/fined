import streamlit as st
from chat_pipeline import run_pipeline

# Configuração da página do Streamlit
st.set_page_config(page_title="Chatbot com Haystack", page_icon="🤖")
st.title("Sistema Educacional RAG com Haystack")
st.caption("Conectado ao seu pipeline Haystack e Ollama")

# Inicializa o histórico de chat no estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso te ajudar hoje?"}]

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("Qual o seu prompt?"):
    # Adiciona a mensagem do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera e exibe a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Analisando documentos..."):
            # Chama a sua função principal do Haystack.
            # Ela já retorna a string da resposta.
            answer = run_pipeline(prompt)
            
            # Exibe a resposta
            st.markdown(answer)
            
            # Adiciona a resposta do assistente ao histórico
            st.session_state.messages.append({"role": "assistant", "content": answer})