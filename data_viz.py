import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import json
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import altair as alt
from collections import Counter
from itertools import combinations
from wordcloud import WordCloud
import networkx as nx
from konlpy.tag import Okt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
hangul_font_path = './font/AppleSDGothicNeoB.ttf'

# 페이지 설정
st.set_page_config(page_title="네이버 뉴스 분석 대시보드", layout="wide")

# 제목
st.title("📊 네이버 뉴스 API 기반 종합 시각화 분석")

# ==================== 데이터 수집 함수 ====================
@st.cache_data(ttl=3600)
def fetch_naver_data(query, num_data=100, client_id=None, client_secret=None):
    """네이버 뉴스 API를 통해 데이터 수집"""
    if not client_id or not client_secret:
        raise ValueError("Client ID와 Secret을 입력해주세요.")
    
    encText = urllib.parse.quote(query)
    results = []
    display_count = min(100, num_data)
    
    for start in range(1, num_data + 1, display_count):
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display={display_count}&sort=date"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        
        try:
            response = urllib.request.urlopen(request)
            rescode = response.getcode()
            if rescode == 200:
                response_body = response.read()
                response_dict = json.loads(response_body.decode('utf-8'))
                results.extend(response_dict.get('items', []))
            else:
                st.error(f"API 요청 오류: {rescode}")
                break
        except Exception as e:
            st.error(f"API 통신 오류: {e}")
            break
        time.sleep(0.05)
    
    df = pd.DataFrame(results)
    if 'title' in df.columns:
        remove_tags = re.compile(r'<.*?>')
        df['title'] = df['title'].apply(lambda x: re.sub(remove_tags, '', x))
        df['description'] = df['description'].apply(lambda x: re.sub(remove_tags, '', x))
        df['pubDate'] = pd.to_datetime(df['pubDate'], format="%a, %d %b %Y %H:%M:%S +0900")
        return df.head(num_data)
    return pd.DataFrame()

# ==================== 사이드바 설정 ====================
st.sidebar.header("⚙️ 분석 설정")

# API 키 입력
client_id = st.sidebar.text_input("네이버 Client ID", type="password")
client_secret = st.sidebar.text_input("네이버 Client Secret", type="password")

# 검색 쿼리 설정
st.sidebar.subheader("검색 설정")
base_query = st.sidebar.text_input("기본 검색어", "K-POP")
num_data = st.sidebar.slider("수집할 뉴스 수", 20, 100, 50)

# 5가지 요인별 추가 키워드
factor_queries = {
    '성별': st.sidebar.text_input("성별 키워드", "남성 팬 OR 여성 팬"),
    '지역': st.sidebar.text_input("지역 키워드", "지역 OR 콘서트"),
    '외국인': st.sidebar.text_input("외국인 키워드", "해외 OR 글로벌"),
    '연령': st.sidebar.text_input("연령 키워드", "MZ세대 OR 10대"),
    '학력': st.sidebar.text_input("학력 키워드", "세계관 OR 철학적")
}

# 분석 파라미터
st.sidebar.subheader("분석 파라미터")
min_word_len = st.sidebar.slider("최소 단어 길이", 2, 4, 2)
min_count_network = st.sidebar.slider("네트워크 최소 빈도", 2, 10, 3)
stopwords_input = st.sidebar.text_area("제외할 단어 (쉼표 구분)", "멤버,그룹,노래,곡,팬덤")

# 분석 실행 버튼
run_analysis = st.sidebar.button("🚀 분석 실행", type="primary")

# ==================== 분석 실행 ====================
if run_analysis:
    if not client_id or not client_secret:
        st.error("⚠️ 네이버 API 키를 입력해주세요!")
    else:
        with st.spinner("데이터 수집 및 분석 중..."):
            # 불용어 설정
            custom_stopwords = set([s.strip() for s in stopwords_input.split(',')])
            okt = Okt()
            
            # 5가지 요인별 데이터 수집
            factor_data = {}
            
            for factor, add_keyword in factor_queries.items():
                query = f"{base_query} {add_keyword}" if add_keyword else base_query
                
                # 데이터 수집
                df_factor = fetch_naver_data(query, num_data, client_id, client_secret)
                
                if df_factor.empty:
                    st.warning(f"{factor} 요인 데이터 수집 실패")
                    continue
                
                # 텍스트 전처리
                all_text = ' '.join(df_factor['title'].tolist() + df_factor['description'].tolist())
                all_text = re.sub(r'[^가-힣A-Za-z\s]', ' ', all_text)
                nouns = okt.nouns(all_text)
                
                # 단어 필터링
                final_nouns = [n for n in nouns if len(n) >= min_word_len and n not in custom_stopwords]
                word_counts = Counter(final_nouns)
                
                # 네트워크 그래프 생성
                node_data = []
                doc_texts = df_factor['title'].tolist() + df_factor['description'].tolist()
                
                for doc in doc_texts:
                    doc_nouns = [n for n in okt.nouns(re.sub(r'[^가-힣A-Za-z\s]', ' ', doc)) 
                                if len(n) >= min_word_len and n not in custom_stopwords]
                    node_data.extend(combinations(sorted(set(doc_nouns)), 2))
                
                edge_counts = Counter(node_data)
                G = nx.Graph()
                
                for edge, weight in edge_counts.items():
                    if weight >= min_count_network:
                        G.add_edge(edge[0], edge[1], weight=weight)
                
                factor_data[factor] = {
                    'df': df_factor,
                    'word_counts': word_counts,
                    'graph': G
                }
            
            st.session_state['factor_data'] = factor_data
            st.success("✅ 분석 완료!")

# ==================== 시각화 ====================
if 'factor_data' in st.session_state and st.session_state['factor_data']:
    factor_data = st.session_state['factor_data']
    
    st.markdown("---")
    
    # 1. Plotly - 요인별 기사 수 비교
    st.header("1️⃣ 요인별 뉴스 기사 수 비교 (Plotly)")
    factor_counts = {k: len(v['df']) for k, v in factor_data.items()}
    df_counts = pd.DataFrame(factor_counts.items(), columns=['요인', '기사 수'])
    
    fig_plotly = px.bar(
        df_counts, 
        x='요인', 
        y='기사 수',
        title='요인별 수집된 뉴스 기사 수',
        color='요인',
        template='plotly_white',
        text='기사 수'
    )
    fig_plotly.update_traces(textposition='outside')
    st.plotly_chart(fig_plotly, use_container_width=True)
    
    st.markdown("---")
    
    # 2. WordCloud - 요인별 핵심 키워드
    st.header("2️⃣ 요인별 핵심 키워드 워드클라우드")
    
    cols = st.columns(len(factor_data))
    for i, (factor, data) in enumerate(factor_data.items()):
        with cols[i]:
            st.subheader(f"{factor}")
            word_counts = data['word_counts']
            
            if word_counts:
                wc = WordCloud(
                    font_path='malgun.ttf',
                    max_words=30,
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_counts)
                
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("키워드 데이터 부족")
    
    st.markdown("---")
    
    # 3. Seaborn - 상위 키워드 빈도 분석
    st.header("3️⃣ 전체 상위 키워드 빈도 분석 (Seaborn)")
    
    all_keywords = Counter()
    for factor_info in factor_data.values():
        all_keywords.update(factor_info['word_counts'])
    
    top_keywords = all_keywords.most_common(20)
    df_keywords = pd.DataFrame(top_keywords, columns=['키워드', '빈도'])
    
    fig_sns, ax_sns = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_keywords, x='빈도', y='키워드', palette='coolwarm', ax=ax_sns)
    ax_sns.set_title('상위 20개 키워드 빈도', fontsize=16, fontweight='bold')
    ax_sns.set_xlabel('빈도', fontsize=12)
    ax_sns.set_ylabel('키워드', fontsize=12)
    st.pyplot(fig_sns)
    plt.close()
    
    st.markdown("---")
    
    # 4. Altair - 키워드 분포 스캐터 플롯
    st.header("4️⃣ 키워드 빈도 vs 중요도 분석 (Altair)")
    
    df_scatter = pd.DataFrame(all_keywords.most_common(50), columns=['키워드', '빈도'])
    df_scatter['중요도'] = df_scatter['빈도'].rank(method='max')
    df_scatter['크기'] = df_scatter['빈도'] * 10
    
    chart = alt.Chart(df_scatter).mark_circle().encode(
        x=alt.X('빈도:Q', title='빈도'),
        y=alt.Y('중요도:Q', title='중요도 (순위)'),
        size=alt.Size('크기:Q', legend=None),
        color=alt.Color('빈도:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['키워드', '빈도', '중요도']
    ).properties(
        title='키워드 빈도 및 중요도 분포',
        width=700,
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("---")
    
    # 5. NetworkX - 키워드 관계망 (외국인 요인)
    st.header("5️⃣ 키워드 관계 네트워크 (외국인 요인)")
    
    if '외국인' in factor_data:
        G = factor_data['외국인']['graph']
        
        if G.number_of_nodes() > 0:
            st.info(f"노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
            
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            node_sizes = [G.degree(node) * 500 for node in G.nodes()]
            edge_widths = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]
            
            fig_net, ax_net = plt.subplots(figsize=(14, 12))
            nx.draw_networkx(
                G, pos,
                with_labels=True,
                node_size=node_sizes,
                width=edge_widths,
                font_size=9,
                node_color='lightblue',
                edge_color='gray',
                alpha=0.7,
                ax=ax_net
            )
            ax_net.set_title("외국인 요인 키워드 관계 네트워크", fontsize=18, fontweight='bold')
            ax_net.axis('off')
            st.pyplot(fig_net)
            plt.close()
        else:
            st.warning("네트워크 생성에 충분한 데이터가 없습니다. 최소 빈도를 낮춰보세요.")
    else:
        st.warning("외국인 요인 데이터가 없습니다.")
    
    st.markdown("---")
    
    # 분석 결과 요약
    st.header("📌 분석 결과 요약")
    st.markdown(f"""
    - **분석된 요인 수**: {len(factor_data)}개
    - **총 수집 기사 수**: {sum(len(v['df']) for v in factor_data.values())}건
    - **전체 고유 키워드 수**: {len(all_keywords)}개
    - **네트워크 노드 수**: {G.number_of_nodes() if '외국인' in factor_data else 0}개
    """)

else:
    st.info("👈 좌측 사이드바에서 설정을 완료하고 '분석 실행' 버튼을 눌러주세요.")
    
    st.markdown("""
    ### 📖 사용 방법
    1. **네이버 API 키 입력**: [네이버 개발자센터](https://developers.naver.com/)에서 발급받은 키를 입력하세요
    2. **검색어 설정**: 분석하고 싶은 주제의 키워드를 입력하세요
    3. **요인별 키워드**: 5가지 요인에 대한 추가 검색어를 설정하세요
    4. **분석 실행**: 버튼을 클릭하여 데이터 수집 및 시각화를 시작하세요
    
    ### 📊 제공되는 시각화
    - **Plotly**: 인터랙티브 막대 그래프
    - **WordCloud**: 키워드 워드클라우드
    - **Seaborn**: 상위 키워드 빈도 차트
    - **Altair**: 키워드 분포 스캐터 플롯
    - **NetworkX**: 키워드 관계 네트워크
    """)
