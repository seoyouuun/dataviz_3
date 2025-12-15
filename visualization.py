if __name__ == "__main__":

    # 1. 사이드바 위젯 (5개 이상 위젯 및 상호작용성 확보)
    base_query = st.sidebar.text_input("1. 그룹 기본 검색어:", 'K-POP 데몬 헌터스', key='base_query')
    num_data_per_factor = st.sidebar.slider("2. 요인별 수집 뉴스 수 (최대 100):", 20, 100, 50, key='num_data_per_factor')
    min_count_network = st.sidebar.number_input("3. 네트워크 최소 빈도:", 3, 10, 5, key='min_count_network')
    stopwords_custom = st.sidebar.text_area("4. 추가 불용어 (쉼표 구분):", '멤버, 그룹, 노래, 곡, 팬덤', key='stopwords_custom')
    min_word_len = st.sidebar.slider("5. 최소 단어 길이:", 2, 4, 2, key='min_word_len')
    run_analysis = st.sidebar.button("✨ 5대 팬덤 요인 분석 실행")
    
    st.session_state['min_count_network'] = min_count_network

    if run_analysis:
        with st.spinner('5대 팬덤 요인별 데이터 수집 및 분석 중...'):
            
            # [4.1] 5가지 요인별 검색 쿼리 정의
            analysis_queries = {
                '성별': f"{base_query} '남성 팬' OR '여성 팬' OR '군대'",
                '지역': f"{base_query} '지역' OR '콘서트 투어'",
                '외국인': f"{base_query} '해외 반응' OR '글로벌' OR '빌보드'",
                '연령별': f"{base_query} 'MZ세대' OR '10대' OR '부모님'",
                '학력수준': f"{base_query} '세계관 해석' OR '철학적' OR '이론'"
            }
            custom_stopwords_set = set([s.strip() for s in stopwords_custom.split(',')])

            factor_data = {}
            for factor, query in analysis_queries.items():
                df_factor = fetch_naver_data(query, num_data_per_factor)
                factor_data[factor] = process_analysis_data(df_factor, min_count_network, min_word_len, custom_stopwords_set)

            st.session_state['analysis_data'] = factor_data
        st.success("✅ 5대 팬덤 형성 요인 분석 완료!")

    
    # 2. 시각화 모듈 호출
    if st.session_state.get('analysis_data'):
        render_dashboard()
