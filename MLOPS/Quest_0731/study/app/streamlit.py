import streamlit as st
import requests
import json

st.title("This is my fist toy project")
st.subheader("This is a subheader")

x = st.text_area("x의 숫자를 입력하세요")
operator = st.selectbox("부등호", ("+", "-", "*", "/"))
y = st.text_area("y의 숫자를 입력하세요")

info_dict = {"x":x, "y": y, "operator": operator}

if st.button("Calculate"):
    result = requests.post(
        url = "http://localhost:8000/calculator",
        data = json.dumps(info_dict),
        verify=False,
    )

    st.subheader(f"result: {result.text}")