import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="데몬헌터스 대시보드",
    page_icon="👥",
    layout="wide"
)

# 메인 화면 제목
st.title("🏠 환영합니다!")

st.markdown("""
안녕하세요!  

C321081 김서연의 페이지입니다.

사이드바를 열고 페이지를 선택해 주세요.
""")

st.divider()

# 마무리
st.caption("Created by C321081 김서연")
