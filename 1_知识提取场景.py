import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.title("知识提取场景")

# 侧边栏选择配置项
st.sidebar.title("配置选项")
llm_options = ['Llama3', 'Qwen2.5', "GPT4o",]
selected_llm = st.sidebar.radio("选择基座大模型", options=llm_options, key="llm_option")
vector_db_options = ['Qdrant', 'Chroma', "Milvus",]
selected_vdb = st.sidebar.radio("选择向量数据库", options=vector_db_options, key="vdb_option")

# todo 根据不同配置选项调整后端配置
if selected_llm == "Llama3":
    pass
if vector_db_options == "Qdrant":
    pass

# 上传文件
uploaded_file = st.file_uploader("请上传文件", accept_multiple_files=False)


# todo 根据上传的文件建立知识图谱
def build_knowledge_graph(uploaded_file):
    # 创建知识图谱（图）
    G = nx.Graph()
    # 添加节点和边（假设这些是从文档中抽取的实体及其关系）
    G.add_node("Alice", type="Person")
    G.add_node("Bob", type="Person")
    G.add_node("Car", type="Object")
    G.add_edge("Alice", "Bob", relation="knows")
    G.add_edge("Alice", "Car", relation="drives")
    G.add_edge("Bob", "Car", relation="owns")
    # 使用matplotlib来绘制图
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 节点位置
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold")
    labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # 显示图
    st.pyplot(plt)

# todo 根据上传的文件提取实体
def extract_entities(uploaded_file):
    data = {
        "Entity Type": ["PERSON", "ORG", "GPE", "DATE", "MONEY", "LOC", "NORP", "WORK_OF_ART"],
        "Count": [15, 7, 9, 5, 3, 6, 4, 2]
    }
    df = pd.DataFrame(data)
    st.subheader("实体类型统计")
    st.dataframe(df)
    # 展示柱状图
    st.subheader("实体类型柱状图")
    fig, ax = plt.subplots()
    ax.bar(df["Entity Type"], df["Count"], color='skyblue')
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('Entity Type Frequency')
    st.pyplot(fig)
    # 展示饼图
    st.subheader("实体类型饼图")
    fig2, ax2 = plt.subplots()
    ax2.pie(df["Count"], labels=df["Entity Type"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax2.set_title('Entity Type Distribution')
    st.pyplot(fig2)

# todo 根据上传的文件提取摘要
def extract_summary(uploaded_file):
    summary = "This is a summary of the uploaded document."
    st.write(summary)


with st.expander("构建知识图谱"):
    with st.spinner("正在生成知识图谱..."):
        build_knowledge_graph(uploaded_file)

with st.expander("各类实体统计"):
    with st.spinner("正在统计实体数量..."):
        extract_entities(uploaded_file)

with st.expander("提取文档摘要"):
    with st.spinner("正在生成文档摘要..."):
        extract_summary(uploaded_file)
