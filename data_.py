#데이터 수집
@st.cache_data(ttl=3600)
def fetch_naver_data(query, num_data=100, client_id=CLIENT_ID, client_secret=CLIENT_SECRET):
    """지정된 쿼리로 네이버 뉴스 데이터를 수집합니다."""
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