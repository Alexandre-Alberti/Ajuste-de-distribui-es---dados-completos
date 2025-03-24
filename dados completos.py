# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:00:50 2025

@author: alexa
"""

import streamlit as st
import numpy as np
from scipy.stats import weibull_min, expon, ks_2samp

# Função para ajustar distribuições
def ajustar_distribuicoes(dados):
    # Convertendo os dados para numpy array
    dados = np.array(dados)
    
    # Ajuste da distribuição Weibull
    c, loc, scale = weibull_min.fit(dados, floc=0)  # floc=0 fixa o parâmetro de localização
    
    # Ajuste da distribuição Exponencial
    loc_exp, scale_exp = expon.fit(dados, floc=0)
    
    # Teste de Kolmogorov-Smirnov
    weibull_dados = weibull_min.rvs(c, loc, scale, size=len(dados))
    expon_dados = expon.rvs(loc_exp, scale_exp, size=len(dados))
    
    ks_weibull, p_weibull = ks_2samp(dados, weibull_dados)
    ks_expon, p_expon = ks_2samp(dados, expon_dados)
    
    return (c, scale, ks_weibull, p_weibull), (scale_exp, ks_expon, p_expon)

# Interface do Streamlit
st.title("Ajuste de Distribuições de Tempo até Falha")

# Entrada de dados pelo usuário
dados_input = st.text_area("Insira os tempos até a falha (use ponto como separador decimal, e separe os números com vírgula):")

if st.button("Ajustar distribuições de probabilidade"):
    try:
        # Processamento dos dados
        dados = list(map(float, dados_input.split(",")))
        
        # Ajuste das distribuições
        (c_weibull, scale_weibull, ks_weibull, p_weibull), (scale_expon, ks_expon, p_expon) = ajustar_distribuicoes(dados)
        
        # Exibir resultados
        st.subheader("Resultados")
        st.write(f"**Distribuição Weibull:** parâmetro de forma = {c_weibull:.4f}, parâmetro de escala = {scale_weibull:.4f}")
        st.write(f"Teste de aderência (KS): p-valor = {p_weibull:.4f}")
        if p_weibull > 0.05:
            st.write("A distribuição Weibull pode ser um bom ajuste para os dados (p > 0.05).")
        else:
             st.write("A distribuição Weibull pode não ser um bom ajuste para os dados (p < 0.05).")
    
        st.write(f"**Distribuição Exponencial:** tempo médio entre falhas = {scale_expon:.4f}")
        st.write(f"Teste de aderência (KS): p-valor = {p_expon:.4f}")
        if p_expon > 0.05:
            st.write("A distribuição Exponencial pode ser um bom ajuste para os dados (p > 0.05).")
        else:
            st.write("A distribuição Exponencial pode não ser um bom ajuste para os dados (p < 0.05).")
        
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {e}")
