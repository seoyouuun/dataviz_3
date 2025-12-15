def process_analysis_data(df_factor, min_count_network, min_word_len, custom_stopwords):
    """단일 요인에 대한 전처리 및 NetworkX 그래프 생성을 수행합니다."""
    # [전처리 및 NetworkX 생성 로직은 1번 파일과 동일하므로 생략]
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