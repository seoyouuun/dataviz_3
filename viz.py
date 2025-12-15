import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import networkx as nx

# 폰트 경로 (main.py에서 동기화됨)
HAN_FONT_PATH = './font/AppleSDGothicNeoB.ttf'


def render_dashboard():

    analysis_data = st.session_state.get('analysis_data', {})
    
    if not analysis_data:
        st.warning("분석 데이터가 로드되지 않았습니다. 사이드바에서 '5대 팬덤 요인 분석 실행'을 클릭해주세요.")
        return

    factor_list = list(analysis_data.keys())
    min_count_network = st.session_state.get('min_count_network', 5) 

    # Plotly (Bar Chart) 
    st.header("1. 요인별 정보량 집중도 비교 (Plotly)")
    st.markdown("수집된 기사 수를 통해 각 요인에 대한 **온라인 관심의 상대적 크기**를 파악합니다.")
    
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


    # 3.2. 핵심 키워드 비교: Matplotlib/WordCloud & NetworkX
    st.header("2. 핵심 키워드 및 연관성 분석 (WordCloud & NetworkX)")
    st.markdown("각 요인별로 가장 중요하게 언급되는 키워드(WordCloud)와 이들의 관계(NetworkX)를 분석합니다.")
    
    
    # wordcloud
    st.subheader("2.1. 요인별 핵심 키워드 (WordCloud)")
    
    wc_cols = st.columns(len(factor_list))
    for i, factor in enumerate(factor_list):
        with wc_cols[i]:
            st.caption(f"**{factor}**")
            data = analysis_data[factor]['word_counts']
            
            if data:
                # WordCloud 객체 생성 (폰트 경로 지정)
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
    st.subheader("2.2. '외국인' 요인 키워드 관계망 (NetworkX)")
    
    G_foreign = analysis_data['외국인']['graph']
    
    if G_foreign.number_of_nodes() > 0:
        # NetworkX 시각화 (폰트 경로 지정)
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
            # Matplotlib의 폰트 설정을 따르므로 font_family에 폰트 이름 대신 경로를 사용하는 경우도 있으나,
            # 여기서는 Matplotlib 설정된 이름을 따름 (AppleSDGothicNeoB가 Matplotlib에 등록된 것으로 가정)
            font_family=plt.rcParams['font.family'], 
            ax=ax_net
        )
        ax_net.set_title("외국인 요인 키워드 관계망 (NetworkX - Matplotlib)", size=15)
        ax_net.axis('off')
        st.pyplot(fig_net)
    else:
        st.warning(f"외국인 요인에 대한 네트워크 생성이 어렵습니다. 최소 빈도({min_count_network})를 낮춰보세요.")


    # 3.4. 키워드 관계 분석: Altair (Scatter Plot) - 요구사항 2 충족
    st.header("3. 키워드 중요도 및 빈도 분석 (Altair)")
    
    all_keywords = Counter()
    for factor in factor_list:
        all_keywords.update(analysis_data[factor]['word_counts'])
    
    df_keywords = pd.DataFrame(all_keywords.most_common(50), columns=['Keyword', 'Frequency'])
    df_keywords['Importance'] = df_keywords['Frequency'].rank(method='max') 
    
    if not df_keywords.empty:
        # Altair는 웹 기반이므로 Matplotlib 폰트 경로 설정의 영향을 받지 않습니다.
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
        
    
    # 4. 결론 및 해석 
    st.markdown("---")
    st.header("📝 종합 분석 해석 및 결론")
    st.markdown("""
        **1. 주요 인사이트 (요인별 집중도 해석):**
        * **Plotly Bar Chart 해석:** 만약 '외국인' 요인의 기사 수가 압도적으로 높다면, 그룹의 팬덤 형성이 **글로벌 인지도와 해외 시장의 성공**에 가장 크게 의존하고 있음을 의미합니다.
        * **WordCloud 해석:** '연령별' 요인에서 '성장'이나 '공감'이 핵심 키워드로 나온다면, 해당 연령대의 팬덤은 그룹과의 **정서적 연결 및 서사 공유**를 중요하게 여긴다는 증거입니다.
        
        **2. 팬덤 형성의 핵심 요인 (Altair 해석):**
        * **Altair Scatter Plot 해석:** **오른쪽 상단**에 위치한 키워드일수록 **빈도(대중적 관심)와 중요도(분석적 중요성)**가 모두 높습니다. 이는 그룹이 반드시 유지하고 강화해야 할 **'균형 잡힌 팬덤 형성 요인'**입니다. 이 키워드들을 중심으로 향후 콘텐츠 전략을 수립해야 합니다.
    """)