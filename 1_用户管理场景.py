import streamlit as st
# 模拟用户管理功能，使用session_state.current_user来保持登录状态，未登录或权限不够不能使用某些功能，比如注册功能只有管理员可以操作

st.markdown("""
    <style>
        .stMessage {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
        }
    </style>
""", unsafe_allow_html=True)

users_db = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user1': {'password': 'user123', 'role': 'user'}
}

# 全局登录状态
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None


# 登录功能
def login(username, password):
    if username in users_db and users_db[username]['password'] == password:
        st.session_state['current_user'] = username
        st.success(f"登录成功！欢迎，{username}")
        return True
    else:
        st.error("用户名或密码错误！")
        return False


# 注销功能
def logout():
    if st.session_state['current_user'] is None:
        st.error("您还未登录！")
        return
    else:
        st.session_state['current_user'] = None
        st.info("您已成功注销！")


# 注册功能（仅管理员可以注册）
def register(username, password):
    if st.session_state['current_user'] == 'admin':
        if username in users_db:
            st.error("该用户名已存在！")
        else:
            users_db[username] = {'password': password, 'role': 'user'}
            st.success(f"用户 {username} 注册成功！")
    else:
        st.error("注册新用户请联系管理员！")


st.title("用户管理场景")

username = st.text_input("用户名")
password = st.text_input("密码", type='password')
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    if st.button("注册"):
        register(username, password)
with col2:
    if st.button("登录"):
        login(username, password)
with col3:
    if st.button("注销"):
        logout()
