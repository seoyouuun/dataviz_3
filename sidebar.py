import streamlit as st

st.set_page_config(
  page_title="김서연의 Streamlit",
  page_icon="⚪️",
  layout="wide", 
  initial_sidebar_state="expanded",
  menu_items={
  '설명서': "https://docs.streamlit.io"
   }
)

#사이드바 설정
st.sidebar.title('다양한 사이드바 위젯들')

st.sidebar.checkbox('외국인 포함')
st.sidebar.divider()
st.sidebar.radio('성별', ['전체', '남성', '여성'])
st.sidebar.slider('나이', 0, 100, (20, 50))
st.sidebar.selectbox('지역', ['서울', '경기', '인천', '대전', '대구', '부산', '광주'])
st.sidebar.selectbox('소득층', ['최하위층', '하위층', '중하위층', '중간층', '중상위층', '상위층', '최상위층'])
st.sidebar.selectbox('학력', ['초졸', '중졸', '고졸', '대졸', '박사 이상'])
