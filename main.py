import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import seaborn as sns
import re
import time
from datetime import datetime
import urllib.request
import json
from itertools import combinations
from konlpy.tag import Okt



# 페이지 기본 설정
st.set_page_config(
    page_title="데몬헌터스 대시보드",
    page_icon="👥",
    layout="wide"
)

# 메인 화면 제목
st.title("환영합니다!")

st.markdown("""
안녕하세요!  

C321081 김서연의 페이지입니다.

사이드바를 열고 페이지를 선택해 주세요.
""")

st.divider()

# 하단
st.caption("Created by C321081 김서연")


#사이드바 설정
import streamlit as st

st.set_page_config(
  page_title="김서연의 Streamlit",
  page_icon="⚪️",
  layout="wide", 
  initial_sidebar_state="expanded",
  menu_items={
  'About': "https://docs.streamlit.io"
   }
)

st.sidebar.title('다양한 사이드바 위젯들')

st.sidebar.checkbox('외국인')
st.sidebar.divider()
st.sidebar.radio('성별', ['전체', '남성', '여성'])
st.sidebar.slider('나이', 0, 100, (20, 30))
st.sidebar.selectbox('지역', ['서울', '경기', '인천', '대전', '대구', '부산', '광주'])
st.sidebar.selectbox('소득층', ['최하위층', '하위층', '중하위층', '중간층', '중상위층', '상위층', '최상위층'])
st.sidebar.selectbox('학력', ['초졸', '중졸', '고졸', '대졸', '박사 이상'])


# 네이버 API 키 (제시된 정보)
CLIENT_ID = 'Hl5maeWyGFS0SOj9hJQt'
CLIENT_SECRET = 'sYYE75Wqpv'

import data_viz

CLIENT_ID = 'Hl5maeWyGFS0SOj9hJQt'
CLIENT_SECRET = 'sYYE75Wqpv'


data_.fetch_naver_data("팬덤", client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

import data_
import viz
