import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import re

# Configurações do Streamlit
st.set_page_config(page_title="Assistente Experience 🤖", page_icon="🤖")
st.title("Assistente Experience 🤖")

model_class = "hf_hub"  # "hf_hub", "openai", "ollama"

# Função para usar o Hugging Face com o token embutido
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.7, max_new_tokens=1024):
    huggingface_token = "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"  # Token adicionado diretamente
    return HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=huggingface_token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        return_full_text=False
    )

def model_openai(model="gpt-4o-mini", temperature=0.7):
    return ChatOpenAI(model=model, temperature=temperature)

def model_ollama(model="phi3", temperature=0.7):
    return ChatOllama(model=model, temperature=temperature)

# Função que gera as respostas do modelo
def model_response(user_query, chat_history, model_class):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    if llm is None:
        yield "Erro ao carregar o modelo."
        return

    system_prompt = "Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}."
    language = "português"

    user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" if model_class.startswith("hf") else "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    chain = prompt_template | llm | StrOutputParser()

    response = chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })

    # Acumulando os chunks em uma lista
    chunks = []
    for chunk in response:
        chunks.append(chunk)

    # Juntando todos os chunks em uma única string
    full_response = "".join(chunks)

    # Limpeza final do texto
    full_response = re.sub(r'\s+', ' ', full_response)  # Remove espaços excessivos
    full_response = re.sub(r'\s([.,!?])', r'\1', full_response)  # Remove espaços antes de pontuações
    full_response = full_response.replace("<|eot_id|>", "").strip()  # Remove o token <|eot_id|>

    # Verifique se há repetições e remova-as
    if ",that," in full_response:
        full_response = full_response.split(",that,")[0]  # Remove a parte repetida

    # Devolvendo a resposta final já limpa
    yield full_response

# Inicializando o histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá, sou o assistente virtual Wake! Como posso ajudar você?")]

# Renderizando o histórico de mensagens no Streamlit
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Entrada do usuário no Streamlit
user_query = st.chat_input("Digite sua mensagem aqui...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Processando a resposta do modelo
    response_stream = model_response(user_query, st.session_state.chat_history, model_class)
    response_content = ""  # Variável para acumular os chunks da resposta

    with st.chat_message("AI"):
        response_placeholder = st.empty()  # Placeholder para atualização dinâmica

        for chunk in response_stream:
            print("Chunk recebido:", chunk)  # Log para depuração
            response_content = chunk
            response_placeholder.markdown(response_content)  # Atualizando o texto dinamicamente

    # Adicionando a resposta final ao histórico
    st.session_state.chat_history.append(AIMessage(content=response_content.strip()))