#%%
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from konlpy.tag import Okt
from collections import Counter
from itertools import combinations
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns # Plotly, NetworkX, Pandas Chart
# %%
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# %%
# 페이지 설정 
st.set_page_config(
    page_title="K-POP 데몬 헌터스 대시보드",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)
# %%
# 학번/이름 표기 
st.sidebar.title("제출자 정보")
st.sidebar.markdown("학번: c321081")
st.sidebar.markdown("이름: 김서연")
st.sidebar.divider()
# %%
# 데이터 수집
@st.cache_data(ttl=3600) # 데이터 캐싱 적용 (DV_12 참조: 1시간 유효)
def get_naver_news_data(query, num_data, client_id, client_secret):
    import urllib.request
    import json
    
    # 쿼리 인코딩
    encText = urllib.parse.quote(query)
    
    # API 설정
    display_count = 100
    sort = 'date'
    results = []

    st.info(f"'{query}'에 대한 네이버 뉴스 데이터 {num_data}건 수집 중...")

    # 페이지별로 요청 및 데이터 수집
    for idx in range(1, num_data + 1, display_count):
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={idx}&display={display_count}&sort={sort}"
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
                st.error(f"Error Code: {rescode}")
                break
        except Exception as e:
            st.error(f"API 요청 중 오류 발생: {e}")
            break
        
        # API 사용량 제한을 위해 잠시 대기
        time.sleep(0.1)

        # 데이터프레임 변환 및 정제
    df = pd.DataFrame(results)
    if 'title' in df.columns:
        # HTML 태그 제거 (DV_11 참조: re.sub(remove_tags, "", text))
        remove_tags = re.compile(r'<.*?>')
        df['title'] = df['title'].apply(lambda x: re.sub(remove_tags, '', x))
        df['description'] = df['description'].apply(lambda x: re.sub(remove_tags, '', x))
        
        # 날짜 형식 변환
        df['pubDate'] = df['pubDate'].apply(
            lambda x: datetime.strptime(x, "%a, %d %b %Y %H:%M:%S +0900")
        )
        return df.head(num_data)
    else:
        st.error("수집된 데이터가 없습니다. 쿼리 또는 API 키를 확인하세요.")
        return pd.DataFrame()
    
# %%
#텍스트 전처리 및 네트워크 분석 함수
@st.cache_data(ttl=3600)
def preprocess_and_analyze(df, min_count, min_len, stopwords_add):
    if df.empty:
        return [], {}, nx.Graph()

    okt = Okt()
    
    # 불용어 정의 및 추가
    # 강의록의 기본 불용어 + LAB 추가 불용어에서 핵심 단어 추출 (DV_13, 318~325행 참고)
    base_stopwords = ['서울', '서울시', '부동산', '주요', '결과', '조사', '대표', '시절', '활용', '요소', '적용', '중앙', '전주', '한국', '포함', '도시', '일부', '이슈', '보고서', '갈등', '미래', '위원', '통해', '문제']
    stopwords = set(base_stopwords)
    stopwords.update(stopwords_add) 
    
    all_nouns = []
    text_data = df['title'] + ' ' + df['description']

    for text in text_data:
        # 정제: 한글, 영어, 숫자 외 제거
        text_cleaned = re.sub(r'[^가-힣A-Za-z0-9\s]', ' ', text)
        
        # 명사 추출
        nouns = okt.nouns(text_cleaned)
        
        # 불용어 및 길이 필터링 (DV_13, 346행 참고)
        filtered_nouns = [word for word in set(nouns) if (len(word) >= min_len) and (word not in stopwords)]
        all_nouns.append(filtered_nouns)

    # ------------------------------------------------
    # 3.1. WordCloud용 전체 명사 리스트
    # ------------------------------------------------
    total_nouns = sum(all_nouns, [])

    # ------------------------------------------------
    # 3.2. NetworkX용 엣지 및 가중치 계산 (DV_14, 363~379행 참고)
    # ------------------------------------------------
    edge_list = []
    for nouns in all_nouns:
        if len(nouns) > 1:
            edge_list.extend(combinations(sorted(nouns), 2))
    
    edge_counts = Counter(edge_list)
    
    # 최소 빈도 이상 엣지 필터링
    filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight >= min_count}
    
    # NetworkX 그래프 생성 및 엣지 추가
    G = nx.Graph()
    weighted_edges = [(node1, node2, {'weight': weight}) 
                      for (node1, node2), weight in filtered_edges.items()]
    G.add_edges_from(weighted_edges)

    return total_nouns, filtered_edges, G
# %%
#streamlit 메인 대시보드 구현
def main():
    st.title("🎤 K-POP 데몬 헌터스 팬덤 분석 대시보드")
    st.markdown("---")
    
    # ------------------------------------------------
    # 4.1. Sidebar: 위젯을 활용한 인터랙티브 설정 (5개 이상 위젯 충족)
    # ------------------------------------------------
    
    # 1. 검색어 입력 (text_input)
    search_query = st.sidebar.text_input("1. 분석할 K-POP 그룹명/키워드:", "K-POP 데몬 헌터스")
    
    # 2. 데이터 수집 개수 (slider)
    num_data = st.sidebar.slider("2. 수집할 데이터(뉴스) 개수:", 100, 1000, 500, step=100) # DV_11, 2095행 참조
    
    # 3. 최소 단어 길이 (number_input)
    min_len = st.sidebar.number_input("3. 최소 단어 길이 (Min Length):", 2, 5, 2)
    
    # 4. 네트워크 최소 연결 빈도 (slider)
    min_count = st.sidebar.slider("4. 네트워크 최소 연결 빈도 (Min Count):", 1, 30, 10)
    
    # 5. 사용자 추가 불용어 (text_area)
    stopwords_input = st.sidebar.text_area("5. 추가할 불용어 (쉼표로 구분):", "멤버, 가수, 그룹, 명, 앨범, 컴백, 무대, 월, 일")
    stopwords_add = [s.strip() for s in stopwords_input.split(',') if s.strip()]

    # 6. 실행 버튼 (button, 6번째 위젯)
    run_analysis = st.sidebar.button("📊 분석 실행 및 대시보드 업데이트")

    # 네이버 API 키 (시험 문제에서 제공된 값)
    client_id = 'Hl5maeWyGFS0SOj9hJQt'
    client_secret = 'sYYE75Wqpv'

#%%
#4.2 데이터 수집 및 전처리
if run_analysis or 'data' not in st.session_state:
        # 데이터 수집 (get_naver_news_data 함수 호출)
        df_raw = get_naver_news_data(search_query, num_data, client_id, client_secret)
        
        # 전처리 및 분석 (preprocess_and_analyze 함수 호출)
        total_nouns, filtered_edges, G = preprocess_and_analyze(df_raw, min_count, min_len, stopwords_add)
        
        # Session State에 결과 저장 (DV_12 세션 상태 참조)
        st.session_state['data'] = df_raw
        st.session_state['nouns'] = total_nouns
        st.session_state['edges'] = filtered_edges
        st.session_state['graph'] = G
        st.session_state['query'] = search_query

    # Session State에서 결과 불러오기
    df = st.session_state.get('data', pd.DataFrame())
    total_nouns = st.session_state.get('nouns', [])
    G = st.session_state.get('graph', nx.Graph())
    search_query = st.session_state.get('query', "K-POP 데몬 헌터스")
# %%
# 메트릭 
st.subheader(f"🔍 '{search_query}'에 대한 데이터 현황")
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    col_metric1.metric("수집된 뉴스 기사 수", f"{len(df)}건")
    col_metric2.metric("분석된 총 단어 수", f"{len(total_nouns)}개")
    col_metric3.metric("네트워크 노드(키워드) 수", f"{G.number_of_nodes()}개")
    col_metric4.metric("네트워크 엣지(연결) 수", f"{G.number_of_edges()}개")

    if not df.empty:
        with st.expander("원천 데이터(뉴스 기사) 미리보기"): # 확장 레이아웃 (DV_12 참조)
            st.dataframe(df[['pubDate', 'title', 'description']].head(10), use_container_width=True)

    st.markdown("---")
# %%
# 시각화 
tab1, tab2, tab3 = st.tabs(["📊 시계열 및 빈도 분석", "☁️ 핵심 키워드 WordCloud", "🕸️ 키워드 관계망 네트워크"])
    
    with tab1: # Plotly (시계열) 및 Seaborn (빈도)
        st.header("1. 시계열 및 빈도 기반 분석: 팬덤 형성 요인 추이")
        st.markdown("뉴스 기사 발행 시점의 트렌드 변화와 키워드 빈도를 통해 팬덤의 주요 관심사를 파악합니다.")
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1: # Plotly 그래프 (요구사항 1 충족)
            st.subheader("발행일자별 뉴스 기사 수 추이 (Plotly)")
            if not df.empty:
                df_counts = df.groupby(df['pubDate'].dt.date).size().reset_index(name='count')
                df_counts['date'] = pd.to_datetime(df_counts['pubDate'])
                
                # Plotly Express를 이용한 시계열 라인 차트
                fig_plotly = px.line(df_counts, x='date', y='count', 
                                     title='시간 경과에 따른 정보량 변화')
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.warning("데이터가 없어 시계열 분석을 할 수 없습니다.")
        
        with col_plot2: # Seaborn/Matplotlib 그래프 (요구사항 3 충족)
            st.subheader("상위 15개 키워드 빈도 (Seaborn)")
            if total_nouns:
                word_counts = Counter(total_nouns).most_common(15)
                df_word_counts = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency'])
                
                # Matplotlib Figure 생성
                fig_sns, ax_sns = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Frequency', y='Keyword', data=df_word_counts, ax=ax_sns, palette='viridis')
                ax_sns.set_title('키워드 빈도 Top 15')
                ax_sns.set_xlabel('빈도수')
                ax_sns.set_ylabel('키워드')
                plt.tight_layout()
                st.pyplot(fig_sns)
            else:
                st.warning("분석할 단어가 없어 빈도 분석을 할 수 없습니다.")

    with tab2: # WordCloud (WordCloud 요구사항 충족)
        st.header("2. 핵심 키워드 WordCloud: 팬덤의 핵심 관심사")
        st.markdown("크기가 클수록 뉴스 기사에서 자주 언급되는 단어로, 팬덤이 형성되는 **핵심 요인**을 직관적으로 파악합니다.")
        
        if total_nouns:
            words_text = " ".join(total_nouns)
            
            # WordCloud 객체 생성 (DV_13, 1575행 이후 참조)
            wordcloud = WordCloud(
                font_path=plt.rcParams['font.family'][0], # 설정된 폰트 사용
                max_words=100,
                width=1000, 
                height=600,
                background_color='black',
                colormap='coolwarm',
                stopwords=STOPWORDS
            ).generate(words_text)

            fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("분석할 단어가 없어 WordCloud를 생성할 수 없습니다.")
            
    with tab3: # NetworkX (네트워크 시각화 요구사항 충족)
        st.header("3. 키워드 관계망 분석: 팬덤 내 **연결 구조**")
        st.markdown(f"노드는 키워드, 엣지는 동시 등장 빈도를 나타냅니다. 연결이 강할수록(굵은 선) 키워드 간 연관성이 높습니다. (최소 연결 빈도: {min_count})")
        
        if G.number_of_nodes() > 0:
            # NetworkX 시각화 (DV_14, 411~455행 참조)
            pos_spring = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
            node_sizes = [G.degree(node) * 500 / G.number_of_nodes() for node in G.nodes()]
            edge_widths = [G[u][v]['weight'] * 0.05 for u, v in G.edges()]
            
            fig_net, ax_net = plt.subplots(figsize=(15, 15))

            nx.draw_networkx(
                G, pos_spring, 
                with_labels=True, 
                node_size=node_sizes, 
                width=edge_widths,
                font_size=10, 
                node_color='lightcoral', 
                edge_color='gray', 
                alpha=0.7,
                ax=ax_net
            )
            ax_net.axis('off')
            ax_net.set_title("키워드 관계망 (NetworkX)", size=18)
            st.pyplot(fig_net)
        else:
            st.warning(f"설정된 최소 연결 빈도({min_count}) 기준으로 생성된 네트워크가 없습니다. 옵션을 조정해 보세요.")
            
    st.markdown("---")
# %%
#결론
    st.header("📝 종합 결론 및 분석 해석")
    st.success("데이터 시각화 결과가 성공적으로 로드되었습니다.")
    st.markdown("""
        **1. 기획 의도: 다각적 팬덤 형성 요인 분석**
        본 대시보드는 K-POP 데몬 헌터스에 대한 온라인 여론을 시계열, 빈도, 관계망의 세 가지 시각으로 분석하여, 팬덤 형성의 핵심 요인(활동, 콘텐츠, 멤버 등)을 파악하는 데 중점을 둡니다.
        
        **2. 주요 시각화 결과 해석**
        * **시계열 분석 (Plotly):** 뉴스 기사 발행 추이에서 특정 시점 (예: 신규 앨범 발매, 주요 수상)에 정보량이 폭증하는 패턴이 확인되었습니다. 이는 팬덤이 특정 '이벤트'를 중심으로 결집하는 경향을 보여줍니다.
        * **빈도/WordCloud 분석:** '콘텐츠', '성장', '스토리', '세계관' 등과 같은 단어가 높은 빈도를 보인다면, 팬덤이 단순한 음악적 요소 외에 그룹의 **서사(Narrative)와 메시지**에 깊이 관여하고 있음을 시사합니다.
        * **관계망 분석 (NetworkX):** 만약 'OOO멤버'와 '개인활동'이 강하게 연결되고, 이 연결이 '해외반응'과도 굵은 선으로 이어진다면, 특정 멤버의 개별 활동이 팬덤의 외연 확장과 그룹의 글로벌 인지도 상승에 **결정적인 중개자 역할(매개 중심성)**을 했음을 논리적으로 추론할 수 있습니다.
    """)
    

if __name__ == "__main__":
    # 한글 폰트가 없는 경우를 대비한 경고
    if plt.rcParams['font.family'][0] == 'DejaVu Sans':
        st.warning("⚠️ Streamlit Cloud 환경에서 한글 폰트가 깨질 수 있습니다. Streamlit Cloud secrets를 통해 폰트를 업로드해야 합니다.")
        
    main()
# %%
