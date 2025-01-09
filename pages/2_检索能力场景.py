import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.html(
"""
<style>
[data-testid='stBaseButton-secondary'] {
  text-indent: -9999px;
  line-height: 0;
}
[data-testid='stBaseButton-secondary']::after {
  line-height: initial;
  content: "点击上传";
  text-indent: 0;
}
[data-testid='stFileUploaderDropzoneInstructions'] > div > small {
  display: none;
}
[data-testid='stFileUploaderDropzoneInstructions'] > div > span {
  display: none;
}
[data-testid='stFileUploaderDropzoneInstructions'] > div::before {
  content: '';
}
</style>
""")
hide_streamlit_style = """
<style>
.stTextArea textarea {
    height: 23px;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("检索能力场景")

# 侧边栏选择配置项
st.sidebar.title("配置选项")
llm_options = ['Llama3', 'Qwen2.5', "GPT4o",]
selected_llm = st.sidebar.radio("选择基座大模型", options=llm_options, key="llm_option")
other_llm_config = st.sidebar.text_input("其他大模型配置")
vector_db_options = ['Qdrant', 'Chroma', "Milvus",]
selected_vdb = st.sidebar.radio("选择向量数据库", options=vector_db_options, key="vdb_option")
other_vdb_config = st.sidebar.text_input("其他向量数据库配置")

# todo 根据不同配置选项调整后端配置
if selected_llm == "Llama3":
    pass
if vector_db_options == "Qdrant":
    pass

# 上传文件
uploaded_file = st.file_uploader("请上传文件")
col1, col2 = st.columns([7.2, 2.8])
with col1:
    user_input = st.text_area("请输入问题")
with col2:
    uploaded_image = st.file_uploader("请上传图片", type=["jpg", "jpeg", "png"])

# todo 对话生成及展示
print(user_input)
print(uploaded_image)
