import streamlit as st
import pandas as pd
import joblib

# Função para gerar datas apenas com dia 01 a partir de 2025
def gerar_datas_primeiro_dia():
    datas = pd.date_range(start='2025-01-01', end='2027-12-01', freq='MS')
    return datas.to_list()

# Função para identificar estação do ano
def estacao_do_ano(data):
    ano = data.year
    primavera = (pd.Timestamp(f'{ano}-09-23'), pd.Timestamp(f'{ano}-12-20'))
    verao     = (pd.Timestamp(f'{ano}-12-21'), pd.Timestamp(f'{ano+1}-03-19'))
    outono    = (pd.Timestamp(f'{ano}-03-20'), pd.Timestamp(f'{ano}-06-20'))
    inverno   = (pd.Timestamp(f'{ano}-06-21'), pd.Timestamp(f'{ano}-09-22'))

    if outono[0] <= data <= outono[1]:
        return 'Outono'
    elif inverno[0] <= data <= inverno[1]:
        return 'Inverno'
    elif primavera[0] <= data <= primavera[1]:
        return 'Primavera'
    elif data >= verao[0] or data <= verao[1]:
        return 'Verão'

# Carrega modelo
modelo = joblib.load('modelo_random_forest.pkl')

# Cria abas
aba = st.tabs(["📈 Previsão", "📄 Sobre o Projeto"])

# -------- Aba de Previsão --------
with aba[0]:
    st.title("POSTECH - Data Analytics - Fase 4")
    st.subheader("Previsão do valor do Petróleo - (FOB)")

    st.sidebar.header("Dados de entrada")

    datas_validas = gerar_datas_primeiro_dia()
    data = st.sidebar.selectbox("Selecione a Data (1º dia de cada mês)", datas_validas)

    producao = st.sidebar.number_input("Produção (MB/d) (ex: 1000)", min_value=100000, max_value=200000, step=1000)

    estacao = estacao_do_ano(pd.to_datetime(data))
    st.sidebar.markdown(f"**Estação do Ano:** {estacao}")

    entrada = {
        "Producao": producao,
        "Estacao_Outono": 1 if estacao == "Outono" else 0,
        "Estacao_Primavera": 1 if estacao == "Primavera" else 0,
        "Estacao_Verão": 1 if estacao == "Verão" else 0
    }
    df_input = pd.DataFrame([entrada])
    
    if st.button("Prever"):
        previsao = modelo.predict(df_input)
        st.success(f"Para a data **{data.strftime('%d/%m/%Y')}** ({estacao}), o valor previsto é: **{previsao[0]:.2f}**")

        # --------- Previsão dos próximos 5 meses ---------
        st.subheader("📅 Previsão para os próximos 5 meses")

        datas_futuras = pd.date_range(start=data + pd.DateOffset(months=1), periods=5, freq='MS')

        lista_entradas = []
        for d in datas_futuras:
            est = estacao_do_ano(d)
            entrada_futura = {
                "Data": d.strftime("%Y-%m-%d"),
                "Estacao": est,
                "Producao": producao,
                "Estacao_Outono": 1 if est == "Outono" else 0,
                "Estacao_Primavera": 1 if est == "Primavera" else 0,
                "Estacao_Verão": 1 if est == "Verão" else 0
            }
            lista_entradas.append(entrada_futura)

        df_futuro = pd.DataFrame(lista_entradas)
        df_futuro_modelo = df_futuro[["Producao", "Estacao_Outono", "Estacao_Primavera", "Estacao_Verão"]]
        previsoes_futuras = modelo.predict(df_futuro_modelo)
        df_futuro["Valor Previsto"] = previsoes_futuras

        st.dataframe(df_futuro[["Data", "Estacao", "Producao", "Valor Previsto"]].style.format({"Valor Previsto": "{:.2f}"}))



# -------- Aba Sobre o Projeto --------
with aba[1]:
    st.title("POSTECH - Data Analytics - Fase 4")

    st.markdown("""
### Problema:

Imagine que você foi escalado como cientista de dados em uma grande empresa de petróleo e precisa criar um modelo preditivo para garantir qual será a previsão do preço do petróleo em dólar e instanciar esse modelo preditivo em uma aplicação para auxiliar na tomada de decisão. Utilize o Streamlit para realizar a interface visual da aplicação e não se esqueça de realizar o deploy do modelo nessa aplicação.  
Base de dados a ser utilizada: [IPEADATA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)

**A entrega deve conter:**
- Link da aplicação do modelo preditivo no Streamlit
- Notebook Python com toda a pipeline de construção do modelo
- Desenvolvimento do objetivo

### Proposta:

Para enriquecer a análise, foi necessário coletar dados complementares, pois utilizar apenas a base de valores do petróleo se mostrou insuficiente para o desenvolvimento eficiente do modelo de machine learning.

Para alcançar o objetivo proposto, será utilizada uma base de dados contendo informações sobre a produção de petróleo por país, incluindo a média diária de barris produzidos.

Com esses dados, será possível cruzar a produção de cada país com o valor médio global do barril de petróleo. O modelo terá como objetivo prever a relação entre produção e preço, ou seja, identificar se uma queda na produção mundial está associada a um aumento no valor do barril — ou, inversamente, se um aumento na produção tende a reduzir seu preço.

### Bases de dados utilizadas:

- Produção mundial de petróleo: [EIA - U.S. Energy Information Administration](https://www.eia.gov)
- Preço por barril do petróleo bruto Brent (FOB) (EIA366_PBRENT366): [IPEADATA](http://www.ipeadata.gov.br)
""")
