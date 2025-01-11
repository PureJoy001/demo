import streamlit as st

st.title("报告生成场景")
# 侧边栏选择配置项
st.sidebar.title("配置选项")
llm_options = ['Llama3', 'Qwen2.5', "GPT4o",]
selected_llm = st.sidebar.radio("选择基座大模型", options=llm_options, key="llm_option")
other_llm_config = st.sidebar.text_input("其他大模型配置")
vector_db_options = ['Qdrant', 'Chroma', "Milvus",]
selected_vdb = st.sidebar.radio("选择向量数据库", options=vector_db_options, key="vdb_option")
other_vdb_config = st.sidebar.text_input("其他向量数据库配置")

uploaded_file = st.file_uploader("请上传文件")

if st.button("生成统计报告"):
    pass

if st.button("生成统计图表"):
    pass