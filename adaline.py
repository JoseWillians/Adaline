import streamlit as st
import numpy as np
import time

# Configuração da página Web
st.set_page_config(page_title="ADALINE - Porta OR", page_icon="🧠", layout="centered")

# Cabeçalho e descrição
st.title("🧠 Rede Neural ADALINE: Porta OR")

# -------------------------------------
# -------  VARIÁVEIS DE ESTADO  -------
# -------------------------------------
if 'treinado' not in st.session_state:
    st.session_state.treinado = False # Para controlar se a rede já foi treinada ou não
    st.session_state.pesos = np.zeros(3) # Para armazenar os pesos finais após o treinamento
    st.session_state.historico = [] # Para armazenar o histórico do EQM a cada época
    st.session_state.epocas_totais = 0 # Para armazenar o número total de épocas até a convergência
    st.session_state.pesos_iniciais = np.zeros(3) # Para armazenar os pesos iniciais antes do treinamento

# -------------------------------------------
# ----- DADOS DE TREINAMENTO (PORTA OR) -----
# -------------------------------------------
X = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1]
])
d = np.array([-1, 1, 1, 1]) # Saídas desejadas para a porta OR (com -1 para falso e 1 para verdadeiro)

eta = 0.01 # Taxa de aprendizado (pode ser ajustada para ver como afeta o treinamento)
epsilon = 1e-5 # Critério de convergência: se a mudança no EQM for menor que epsilon, consideramos que a rede convergiu
max_epocas = 1000 # Limite máximo de épocas para evitar loops infinitos caso a rede não converja

# Função para calcular o Erro Quadrático Médio
def calcular_eqm(w, X, d):
    """Calcula o Erro Quadrático Médio"""
    p = len(d) # Número de padrões de treinamento
    eqm = 0 
    # Para cada padrão de entrada, calcula a saída da rede e acumula o erro quadrático
    for k in range(p):
        u = np.dot(w, X[k])
        eqm += (d[k] - u)**2
    return eqm / p 

# ---------------------------------------------
# -----------  ÁREA DE TREINAMENTO  -----------
# ---------------------------------------------
st.header("1. Treinar o Neurônio")
st.write("Configure como a IA deve nascer e inicie o treinamento.")

# Toggle para escolher se a semente é fixa ou aleatória
usar_semente = st.toggle(
    "Usar Semente Fixa (Resultados Reprodutíveis)", 
    value=True,
    help="Se ativado, a IA sempre começará com os mesmos pesos iniciais. Se desativado, cada treinamento será único e aleatório!"
)

# Botão para iniciar o treinamento
if st.button("Iniciar Treinamento", type="primary"):
    
    # Verifica a escolha do usuário
    if usar_semente:
        np.random.seed(42) # Trava os números aleatórios
    else:
        np.random.seed(None) # Libera para aleatoriedade total baseada no relógio do PC
    
    w = np.random.uniform(-0.1, 0.1, 3) # Pesos iniciais aleatórios entre -0.1 e 0.1
    
    epoca = 0 # Contador de épocas
    historico_eqm = [] # Lista para armazenar o histórico do EQM a cada época
    
    barra_progresso = st.progress(0, text="Treinando...") # Barra de progresso para feedback visual

    # Salva os pesos iniciais para mostrar na tela depois
    st.session_state.pesos_iniciais = w.copy()

    # Loop de treinamento
    while epoca < max_epocas:
        eqm_anterior = calcular_eqm(w, X, d) # Calcula o EQM antes de atualizar os pesos

        # Atualiza a barra de progresso (progresso baseado no número de épocas)
        for k in range(len(d)):
            u = np.dot(w, X[k])
            w = w + eta * (d[k] - u) * X[k]

        # Calcula o EQM após a atualização dos pesos e armazena no histórico 
        eqm_atual = calcular_eqm(w, X, d)
        historico_eqm.append(eqm_atual) 
        epoca += 1
        
        # Atualiza a barra de progresso a cada época
        if abs(eqm_atual - eqm_anterior) <= epsilon:
            break

    barra_progresso.empty() # Remove a barra de progresso após o treinamento
    
    # Salva TODOS os resultados na memória do site antes de mostrar
    st.session_state.pesos = w # Pesos finais após o treinamento
    st.session_state.historico = historico_eqm # Histórico do EQM a cada época
    st.session_state.epocas_totais = epoca # Número total de épocas até a convergência
    st.session_state.treinado = True # Marca que a rede foi treinada para liberar a área de testes

# --- MOSTRAR RESULTADOS ---
if st.session_state.treinado:
    st.success(f"✅ Treinamento concluído com sucesso na época **{st.session_state.epocas_totais}**!")
    
    w_init = st.session_state.pesos_iniciais # Pesos iniciais antes do treinamento
    w_final = st.session_state.pesos # Pesos finais após o treinamento
    
    # Mostra o antes e o depois
    st.write(f"**Pesos Iniciais (Como a IA nasceu):** `[{w_init[0]:.4f}, {w_init[1]:.4f}, {w_init[2]:.4f}]`") # Formata os pesos para mostrar com 4 casas decimais

    st.write(f"**Pesos Finais Ajustados (Após aprender):** `[{w_final[0]:.4f}, {w_final[1]:.4f}, {w_final[2]:.4f}]`") # Formata os pesos para mostrar com 4 casas decimais
    
    st.subheader("📉 Curva de Aprendizado (EQM)")
    st.line_chart(st.session_state.historico) # Mostra a curva do EQM ao longo das épocas para visualizar o processo de aprendizado da rede

st.divider()

# -------------------------------
# -------  ÁREA DE TESTE  -------
# -------------------------------
st.header("2. Fase de Operação (Teste)")

# Verifica se a rede já foi treinada para liberar a área de testes
if st.session_state.treinado:
    st.write("Selecione os valores de entrada para testar se a rede aprendeu a regra do OR:")
    
    # Cria duas colunas para organizar melhor os inputs
    col1, col2 = st.columns(2) # Coluna 1 para X1 e Coluna 2 para X2
    # Cada selectbox tem opções -1 e 1, mas com rótulos mais amigáveis para o usuário entender que 1 é "Verdadeiro" e -1 é "Falso"
    with col1: 
        x1_val = st.selectbox("Entrada 1 (X1):", [-1, 1], format_func=lambda x: "1 (Verdadeiro)" if x==1 else "-1 (Falso)")
    with col2:
        x2_val = st.selectbox("Entrada 2 (X2):", [-1, 1], format_func=lambda x: "1 (Verdadeiro)" if x==1 else "-1 (Falso)")
        
    # Botão para testar a rede com as entradas selecionadas
    if st.button("🧪 Testar Rede"):
        entrada_teste = np.array([-1, x1_val, x2_val]) # Vetor de entrada para teste (com o bias fixo em -1)
        u = np.dot(st.session_state.pesos, entrada_teste) # Calcula a saída da rede para as entradas de teste usando os pesos finais aprendidos
        
        y = 1 if u >= 0 else -1 # A função de ativação é um degrau: se u >= 0, a saída é 1 (Verdadeiro), caso contrário é -1 (Falso)
        
        resultado = "VERDADEIRO" if y == 1 else "FALSO"
        cor = "green" if y == 1 else "red"
        
        st.markdown(f"### A rede respondeu: :{cor}[**{resultado}**]")
        st.caption(f"Cálculo matemático bruto (u): {u:.4f}")

else:
    st.info("Você precisa treinar a rede no Passo 1 para liberar a área de testes.")
