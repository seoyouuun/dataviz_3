import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import networkx as nx

# 폰트 경로
HAN_FONT_PATH = './font/AppleSDGothicNeoB.ttf'


def render_dashboard():

    analysis_data = st.session_state.get('analysis_data', {})
    
    if not analysis_data:
        st.warning("분석 데이터가 로드되지 않았습니다. ")
        return

    factor_list = list(analysis_data.keys())
    min_count_network = st.session_state.get('min_count_network', 5) 

    # Plotly (Bar Chart) 
    st.header("요인별 정보량 집중도 비교 (Plotly)")
    
    factor_counts = {k: len(v['df']) for k, v in analysis_data.items()}
    df_factor_counts = pd.DataFrame(factor_counts.items(), columns=['팬덤 요인', '뉴스 기사 수'])
    
    fig_plotly = px.bar(
        df_factor_counts, 
        x='팬덤 요인', 
        y='뉴스 기사 수', 
        title='요인별 검색 정보량 (뉴스 기사 수)',
        color='팬덤 요인', 
        template='plotly_white'
    )
    st.plotly_chart(fig_plotly, use_container_width=True)


    # 핵심 키워드 비교
    st.header("핵심 키워드 및 연관성 분석")
   
    
    # wordcloud
    st.subheader("인별 핵심 키워드")
    
    wc_cols = st.columns(len(factor_list))
    for i, factor in enumerate(factor_list):
        with wc_cols[i]:
            st.caption(f"**{factor}**")
            data = analysis_data[factor]['word_counts']
            
            if data:
                # WordCloud 객체 생성
                wc = WordCloud(
                    font_path=HAN_FONT_PATH, 
                    max_words=50, 
                    width=300, 
                    height=200, 
                    background_color='white'
                ).generate_from_frequencies(data)
                
                # Matplotlib Fig를 Streamlit에 출력
                fig, ax = plt.subplots(figsize=(3, 2)) 
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("키워드 부족")

    # 3.3. 키워드 관계 분석: NetworkX
    st.subheader("'외국인' 요인 키워드 관계망")
    
    G_foreign = analysis_data['외국인']['graph']
    
    if G_foreign.number_of_nodes() > 0:
        # NetworkX 시각화
        pos_spring = nx.spring_layout(G_foreign, k=0.4, iterations=50, seed=42)
        node_sizes = [G_foreign.degree(node) * 300 for node in G_foreign.nodes()]
        edge_widths = [G_foreign[u][v]['weight'] * 0.2 for u, v in G_foreign.edges()]
        
        fig_net, ax_net = plt.subplots(figsize=(10, 10))
        nx.draw_networkx(
            G_foreign, pos_spring, 
            with_labels=True, 
            node_size=node_sizes, 
            width=edge_widths,
            font_size=10, 
            node_color='lightcoral', 
            edge_color='gray', 
            alpha=0.7,
            font_family=plt.rcParams['font.family'], 
            ax=ax_net
        )
        ax_net.set_title("외국인 요인 키워드 관계망 (NetworkX - Matplotlib)", size=15)
        ax_net.axis('off')
        st.pyplot(fig_net)
    else:
        st.warning(f"외국인 요인에 대한 네트워크 생성이 어렵습니다.")


    # 키워드 관계 분석
    st.header("키워드 중요도 및 빈도 분석")
    
    all_keywords = Counter()
    for factor in factor_list:
        all_keywords.update(analysis_data[factor]['word_counts'])
    
    df_keywords = pd.DataFrame(all_keywords.most_common(50), columns=['Keyword', 'Frequency'])
    df_keywords['Importance'] = df_keywords['Frequency'].rank(method='max') 
    
    if not df_keywords.empty:
        chart = alt.Chart(df_keywords).mark_circle().encode(
            x=alt.X('Frequency', title='빈도 (X축: 대중적 관심)'),
            y=alt.Y('Importance', title='중요도 (Y축: 분석적 중요도)'),
            size='Frequency', 
            color=alt.Color('Frequency', scale=alt.Scale(range='heatmap')),
            tooltip=['Keyword', 'Frequency', 'Importance']
        ).properties(
            title="키워드 빈도 vs. 중요도 (Altair Scatter)"
        ).interactive() 
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("분석할 키워드 데이터가 부족합니다.")
        
    
