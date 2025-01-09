import streamlit as st
import pymupdf4llm
import chromadb
from PIL import Image
from util import extract_image_context
from vanna.utils import deterministic_uuid
import base64
import openai
from llama_index.llms.openai import OpenAI as llamaindex_openai
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from  openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, load_index_from_storage, StorageContext
from prompt_templates import custom_prompt
from llama_index.readers.file import PDFReader
import os
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, Sto
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

st.set_page_config(page_title="mmai chat", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = "sk-aal58s2gWETYNHTiD49904329c7942C7Ac8dC70625041265"
st.title("Chat with your files 💬")
st.info("回答将基于你的文件内容", icon="📃")

Settings.llm = llamaindex_openai(model="gpt-4o", temperature=0.3, system_prompt="你是一位回答用户问题的人工智能助手，由北邮MMAI实验室研发。"
                                                                            "假设所有问题都与对应文档资料资料"
                                                                            "有关。保持你的答案以事实为基础——不要产生幻觉。如果用户的问题是要寻找某些图片或示意图，\
                                                                            我会从其他地方获取图片，因此此时你只需要回答知识库中和用户提问有关的部分，并在最后加上类似“以下是xx示意图”的回答，\
                                                                            我会在你的回答之后加上我从其他流程获取的图片，不要出现类似“无法找到示意图”之类的回答。如果用户提问不涉及\
                                                                            寻找图片相关的意图，你只需要正常根据知识库资料回答用户的问题，用中文回答。",
                                                                  api_base="https://ai-yyds.com/v1", api_key="sk-aal58s2gWETYNHTiD49904329c7942C7Ac8dC70625041265")
Settings.embed_model = OpenAIEmbedding(api_base="https://ai-yyds.com/v1", api_key="sk-aal58s2gWETYNHTiD49904329c7942C7Ac8dC70625041265")
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "向我提问吧!"}
    ]



@st.cache_resource(show_spinner=False)
def load_embed_model(model_path):
    return SentenceTransformer(model_path)
embed_model = load_embed_model("/home/tuxinyu/models/bge-small-zh-v1.5")
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = [embed_model.encode(text, normalize_embeddings=True) for text in texts]
        return embeddings
@st.cache_resource(show_spinner=False)
def get_img_collection():
    chroma_client = chromadb.PersistentClient(
        path="/data/my_chromadb_vector", settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    try:
        chroma_client.delete_collection(name= "demo_img")
        print("Deleted collections ")
    except Exception as e:
        print("Deleted collections failed")
        pass
    img_collection = chroma_client.get_or_create_collection(
        name="demo_img",
        embedding_function=MyEmbeddingFunction()
    )
    return img_collection

img_collection = get_img_collection()

def get_related_img(question: str, distance_threshold: float = 0.5):
    result = img_collection.query(
        query_texts=[question],
        n_results=1,
    )
    try:
        document = result['documents'][0][0]
        metadata = result['metadatas'][0][0]
        distance = result['distances'][0][0]
        return metadata['img_path'] if distance < distance_threshold else '', document
    except Exception as e:
        return '', ''




def chat_with_img(image_path, user_prompt):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image( image_path )

    client = OpenAI(
            base_url="https://ai-yyds.com/v1",
            api_key="sk-aal58s2gWETYNHTiD49904329c7942C7Ac8dC70625041265")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content



@st.cache_resource(show_spinner=False)
def index_raw_data(file_path):
    with st.spinner(text="正在进行文件处理， 这可能需要几分钟时间."):
        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        documents = llama_reader.load_data(file_path)
        # parse imgs

        if not os.path.exists(f"saved_imgs"):
            os.makedirs(f"saved_imgs")
        print('file_path:', file_path)
        md_text = pymupdf4llm.to_markdown(file_path, write_images=True,
                                          image_path=f"saved_imgs")
        image_list = extract_image_context(md_text, 1)
        for image_text_pair in image_list:
            img_path, text = image_text_pair[0], image_text_pair[1]
            print(img_path, text)
            id = deterministic_uuid(img_path) + "-img"
            img_collection.add(
                documents=text,
                metadatas={'img_path': img_path},
                ids=id,
            )
        index = VectorStoreIndex.from_documents(documents)
        return index


#index = load_data()
#index.storage_context.persist("cache/C++语言程序设计_OCR.pdf")


@st.cache_resource(show_spinner=False)
def upload_file_to_local_path(uploaded_file, saved_prefix):
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name
    suffix = file_name.split(".")[-1]
    uploaded_file_path = f'{saved_prefix}/{file_name}'
    print('uploading: ' + uploaded_file_path)
    with open(uploaded_file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return uploaded_file_path



uploaded_pdf = st.file_uploader("Choose your file", accept_multiple_files=False)
uploaded_image = st.file_uploader("请上传图片", type=["jpg", "jpeg", "png"])
uploaded_pdf_path = upload_file_to_local_path(uploaded_pdf, "uploaded_data")
uploaded_img_path = upload_file_to_local_path(uploaded_image, "uploaded_img")
if uploaded_pdf_path:
    chat_index = index_raw_data(uploaded_pdf_path)




# if uploaded_file is not None:
#     index = load_index_from_local_storage(to_be_cached_index_path) if not have_new_file else index_raw_data(uploaded_file_path, to_be_cached_index_path)
#     if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#         st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if uploaded_img_path and not uploaded_pdf_path:
                # simple MLLM QA
                response = chat_with_img(uploaded_img_path, prompt)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history
            elif uploaded_pdf_path and not uploaded_img_path:
                # QA with pdf
                query_engine = chat_index.as_query_engine()
                chat_history = []
                chat_engine = CondenseQuestionChatEngine.from_defaults(
                    query_engine=query_engine,
                    condense_question_prompt=custom_prompt,
                    chat_history=chat_history,
                    verbose=True,
                )
                response = chat_engine.chat(prompt, chat_history).response
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
                # add img
                if '示意图' in prompt or '图片' in prompt:
                    img_path, _ = get_related_img(prompt, 1)
                else:
                    img_path, _ = get_related_img(prompt)
                if img_path:
                    image = Image.open(img_path)
                    st.write(image)
                    message = {"role": "assistant", "content": image}
                    st.session_state.messages.append(message)
            elif uploaded_pdf_path and uploaded_img_path:

                # search img
                img_text = chat_with_img(uploaded_img_path, "我正在做一个图片搜索的任务，因此需要你帮我生成你帮我生成这幅图片对应的文字描述信息，以便我根据文本信息在知识库中搜索。请你直接生成该图片的文字描述")
                img_path, document = get_related_img(img_text, 1)
                print(img_text, document)
                image = Image.open(img_path)
                # search in knowledge
                query_engine = chat_index.as_query_engine()
                chat_history = []
                chat_engine = CondenseQuestionChatEngine.from_defaults(
                    query_engine=query_engine,
                    condense_question_prompt=custom_prompt,
                    chat_history=chat_history,
                    verbose=True,
                )
                prompt += '''
                以下是用户提问中的“图片”的信息，对你回答问题也许有帮助：
                '''
                prompt += document
                response = chat_engine.chat(prompt, chat_history).response
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)



                st.write(image)
                message = {"role": "assistant", "content": image}
                st.session_state.messages.append(message)


