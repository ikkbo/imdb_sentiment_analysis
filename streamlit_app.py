<<<<<<< HEAD
# streamlit_app.py
import streamlit as st
import requests

st.title("IMDB 情感分析 Demo")

# 单条文本输入
text = st.text_area("输入文本", "This movie was amazing!")

if st.button("预测情感"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": text}
    )
    result = response.json()
    st.write(f"情感：{result['label']}")
    st.write(f"置信度：{result['confidence']:.2f}")

# 批量文本输入
batch_texts = st.text_area("批量文本（每行一条）", "I loved it\nNot good")
if st.button("批量预测"):
    texts_list = [line.strip() for line in batch_texts.split("\n") if line.strip()]
    response = requests.post(
        "http://127.0.0.1:8000/predict_batch",
        json={"texts": texts_list}
    )
    results = response.json()
    for r in results:
        st.write(f"{r['text']} → {r['label']} ({r['confidence']:.2f})")
=======
# streamlit_app.py
import streamlit as st
import requests

st.title("IMDB 情感分析 Demo")

# 单条文本输入
text = st.text_area("输入文本", "This movie was amazing!")

if st.button("预测情感"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": text}
    )
    result = response.json()
    st.write(f"情感：{result['label']}")
    st.write(f"置信度：{result['confidence']:.2f}")

# 批量文本输入
batch_texts = st.text_area("批量文本（每行一条）", "I loved it\nNot good")
if st.button("批量预测"):
    texts_list = [line.strip() for line in batch_texts.split("\n") if line.strip()]
    response = requests.post(
        "http://127.0.0.1:8000/predict_batch",
        json={"texts": texts_list}
    )
    results = response.json()
    for r in results:
        st.write(f"{r['text']} → {r['label']} ({r['confidence']:.2f})")
>>>>>>> 9ef5a78 (首次提交)
