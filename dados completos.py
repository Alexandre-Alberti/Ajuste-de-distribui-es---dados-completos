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
dados_input = st.text_area("Insira os tempos até a falha, separados por vírgula:")

if st.button("Ajustar"):
    try:
        # Processamento dos dados
        dados = list(map(float, dados_input.split(",")))
        
        # Ajuste das distribuições
        (c_weibull, scale_weibull, ks_weibull, p_weibull), (scale_expon, ks_expon, p_expon) = ajustar_distribuicoes(dados)
        
        # Exibir resultados
        st.subheader("Resultados do Ajuste")
        st.write(f"**Distribuição Weibull:** c = {c_weibull:.4f}, scale = {scale_weibull:.4f}")
        st.write(f"Teste KS: D = {ks_weibull:.4f}, p-valor = {p_weibull:.4f}")
        
        st.write(f"**Distribuição Exponencial:** scale = {scale_expon:.4f}")
        st.write(f"Teste KS: D = {ks_expon:.4f}, p-valor = {p_expon:.4f}")
        
        # Conclusão sobre aderência
        st.subheader("Conclusão")
        if p_weibull > 0.05:
            st.write("A distribuição Weibull pode ser um bom ajuste para os dados (p > 0.05).")
        else:
            st.write("A distribuição Weibull pode não ser um bom ajuste para os dados (p < 0.05).")
        
        if p_expon > 0.05:
            st.write("A distribuição Exponencial pode ser um bom ajuste para os dados (p > 0.05).")
        else:
            st.write("A distribuição Exponencial pode não ser um bom ajuste para os dados (p < 0.05).")
        
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {e}")
