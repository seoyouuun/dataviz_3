import streamlit as st
import urllib.parse
import json 
import pandas as pd
import urllib.request
import re
import time

#데이터 수집
@st.cache_data(ttl=3600)
# data_.py
def fetch_naver_data(query, num_data=100, client_id=None, client_secret=None):
    if not client_id or not client_secret:
        raise ValueError("Client ID and Secret must be provided.")
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
                st.error(f"API 요청 오류 ({query}): {rescode}")
                break
        except Exception as e:
            st.error(f"API 통신 오류: {e}")
            break
        time.sleep(0.05) 
    
    # 데이터프레임 변환 및 HTML 태그 제거 
    df = pd.DataFrame(results)
    if 'title' in df.columns:
        remove_tags = re.compile(r'<.*?>')
        df['title'] = df['title'].apply(lambda x: re.sub(remove_tags, '', x))
        df['description'] = df['description'].apply(lambda x: re.sub(remove_tags, '', x))
        df['pubDate'] = pd.to_datetime(df['pubDate'], format="%a, %d %b %Y %H:%M:%S +0900")
        return df.head(num_data)
    return pd.DataFrame()

def process_analysis_data(df_factor, min_count_network, min_word_len, custom_stopwords):
    okt = Okt()
    
    all_text = ' '.join(df_factor['title'].tolist() + df_factor['description'].tolist())
    all_text = re.sub(r'[^가-힣A-Za-z\s]', ' ', all_text)
    nouns = okt.nouns(all_text)
    
    final_nouns = [n for n in nouns if len(n) >= min_word_len and n not in custom_stopwords]
    word_counts = Counter(final_nouns)

    node_data = []
    doc_texts = df_factor['title'].tolist() + df_factor['description'].tolist()
    
    for doc in doc_texts:
        doc_nouns = [n for n in okt.nouns(re.sub(r'[^가-힣A-Za-z\s]', ' ', doc)) if len(n) >= min_word_len and n not in custom_stopwords]
        node_data.extend(combinations(sorted(set(doc_nouns)), 2))
        
    edge_counts = Counter(node_data)
    G = nx.Graph()
    
    for edge, weight in edge_counts.items():
        if weight >= min_count_network:
            G.add_edge(edge[0], edge[1], weight=weight)
            
    return {
        'df': df_factor,
        'word_counts': word_counts,
        'graph': G
    }

if __name__ == "__main__":

    # 1. 사이드바 위젯
    base_query = st.sidebar.text_input("1. 그룹 기본 검색어:", 'K-POP 데몬 헌터스', key='base_query')
    num_data_per_factor = st.sidebar.slider("2. 요인별 수집 뉴스 수 (최대 100):", 20, 100, 50, key='num_data_per_factor')
    min_count_network = st.sidebar.number_input("3. 네트워크 최소 빈도:", 3, 10, 5, key='min_count_network')
    stopwords_custom = st.sidebar.text_area("4. 추가 불용어 (쉼표 구분):", '멤버, 그룹, 노래, 곡, 팬덤', key='stopwords_custom')
    min_word_len = st.sidebar.slider("5. 최소 단어 길이:", 2, 4, 2, key='min_word_len')
    run_analysis = st.sidebar.button("5대 팬덤 요인 분석 실행")
    
    st.session_state['min_count_network'] = min_count_network

    if run_analysis:
        with st.spinner('5대 팬덤 요인별 데이터 수집 및 분석 중'):
            
            # [4.1] 5가지 요인별 검색 쿼리 정의
            analysis_queries = {
                '성별': f"{base_query} '남성' OR '여성' OR ",
                '지역': f"{base_query} '도시' OR '지방'",
                '외국인': f"{base_query} '글로벌' OR '국내'",
                '연령별': f"{base_query} 'MZ세대' OR '10대' OR '부모님'",
                '학력수준': f"{base_query} '저학력' OR '고학력'"
            }
            custom_stopwords_set = set([s.strip() for s in stopwords_custom.split(',')])

            factor_data = {}
            for factor, query in analysis_queries.items():
                df_factor = fetch_naver_data(query, num_data_per_factor)
                factor_data[factor] = process_analysis_data(df_factor, min_count_network, min_word_len, custom_stopwords_set)

            st.session_state['analysis_data'] = factor_data
        st.success("팬덤 형성 요인 분석 완료")

    
    # 2. 시각화 모듈 호출
    if st.session_state.get('analysis_data'):
        render_dashboard()

