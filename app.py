import os
import sqlite3
import tempfile
from uuid import uuid4

import streamlit as st

from shopping_agent_assistant import agent
from shopping_agent_assistant.db import ensure_database, DB_PATH
from shopping_agent_assistant.runtime import debug_log

ensure_database()

st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_catalog_stats():
    if not DB_PATH.exists():
        return {"products": 0, "categories": 0, "organic": 0}

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT category), SUM(is_organic) FROM products")
        products, categories, organic = cursor.fetchone()

    return {
        "products": products or 0,
        "categories": categories or 0,
        "organic": organic or 0,
    }


def message_content_to_text(content):
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            text = message_content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)

    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return message_content_to_text(content[key])
        return ""

    return str(content)


def escape_markdown_money(text):
    return message_content_to_text(text).replace("$", r"\$")


def is_image_prompt(content):
    return message_content_to_text(content).startswith("I uploaded a product image")


def submit_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.needs_response = True
    st.rerun()


def render_message(message):
    role = message["role"]
    content = message_content_to_text(message["content"])

    with st.chat_message(role):
        if role == "user" and is_image_prompt(content):
            filename = content.split("Image path:")[-1].strip()
            st.markdown(f"**Image search**: `{os.path.basename(filename)}`")
        else:
            st.markdown(escape_markdown_money(content))


def render_assistant_response():
    with st.chat_message("assistant"):
        with st.spinner("Checking the catalog and comparing options..."):
            st.session_state.session_round += 1
            debug_log(session_round=st.session_state.session_round)

            latest_user_message = st.session_state.messages[-1]
            result = agent.invoke(
                {"messages": [latest_user_message]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            response = message_content_to_text(result["messages"][-1].content).replace("`", "")

            debug_log(assistant_response=response)
        st.markdown(escape_markdown_money(response))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.needs_response = False
    st.session_state.pop("pending_image", None)
    st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "needs_response" not in st.session_state:
    st.session_state.needs_response = False

if "session_round" not in st.session_state:
    st.session_state.session_round = 0

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())


st.markdown(
    """
    <style>
        :root {
            --shop-ink: #18201c;
            --shop-muted: #647067;
            --shop-line: #dbe4dc;
            --shop-panel: #ffffff;
            --shop-panel-soft: #f5f8f4;
            --shop-green: #2f6f4e;
            --shop-mint: #dff2e5;
            --shop-gold: #c7891f;
            --shop-blue: #436f8f;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 10%, rgba(223, 242, 229, 0.72), transparent 24rem),
                linear-gradient(180deg, #fbfcf8 0%, #eef5ee 100%);
            color: var(--shop-ink);
        }

        [data-testid="stSidebar"] {
            background: #f7faf6;
            border-right: 1px solid var(--shop-line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--shop-muted);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 6rem;
            max-width: 1180px;
        }

        .shop-hero {
            border: 1px solid rgba(47, 111, 78, 0.18);
            background:
                linear-gradient(135deg, rgba(47, 111, 78, 0.98), rgba(67, 111, 143, 0.92)),
                linear-gradient(90deg, rgba(255,255,255,0.09) 1px, transparent 1px);
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            color: #ffffff;
            box-shadow: 0 18px 45px rgba(28, 48, 36, 0.15);
        }

        .shop-kicker {
            color: rgba(255, 255, 255, 0.76);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .shop-hero h1 {
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1.02;
            letter-spacing: 0;
            margin: 0 0 0.75rem 0;
        }

        .shop-hero p {
            color: rgba(255, 255, 255, 0.86);
            max-width: 720px;
            margin: 0;
            font-size: 1.02rem;
        }

        .metric-card {
            height: 100%;
            border: 1px solid var(--shop-line);
            background: rgba(255, 255, 255, 0.84);
            border-radius: 8px;
            padding: 1rem;
        }

        .metric-label {
            color: var(--shop-muted);
            font-size: 0.8rem;
            font-weight: 650;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            color: var(--shop-ink);
            font-size: 1.75rem;
            font-weight: 760;
            line-height: 1;
        }

        .section-title {
            color: var(--shop-ink);
            font-size: 1rem;
            font-weight: 750;
            margin: 0.8rem 0 0.25rem 0;
        }

        .section-note {
            color: var(--shop-muted);
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .empty-state {
            border: 1px dashed rgba(47, 111, 78, 0.35);
            background: rgba(223, 242, 229, 0.45);
            border-radius: 8px;
            padding: 1.1rem;
            color: var(--shop-muted);
            margin: 0.55rem 0 0.35rem;
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(47, 111, 78, 0.22);
            background: #ffffff;
            color: var(--shop-ink);
            font-weight: 650;
            min-height: 2.6rem;
            white-space: normal;
            text-align: left;
        }

        .stButton > button:hover {
            border-color: var(--shop-green);
            color: var(--shop-green);
            background: var(--shop-mint);
        }

        [data-testid="stChatMessage"] {
            border-radius: 8px;
            padding: 0.9rem;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 8px;
        }

        .sidebar-card {
            border: 1px solid var(--shop-line);
            background: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.75rem 0 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


stats = load_catalog_stats()

with st.sidebar:
    st.markdown("### Image Match")
    st.caption("Upload a product photo to search the catalog by visual similarity.")

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.image(uploaded_file, width='stretch')

    image_disabled = uploaded_file is None or st.session_state.needs_response
    if st.button(
        "Find similar products",
        width='stretch',
        disabled=image_disabled,
    ):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = (
            "I uploaded a product image. Please analyze it and find similar products "
            f"in the store. Image path: {image_path}"
        )
        st.session_state.pending_image = uploaded_file.name
        submit_prompt(prompt)

    st.divider()
    st.markdown("### Catalog")
    st.markdown(
        f"""
        <div class="sidebar-card">
            <strong>{stats["products"]}</strong> products<br>
            <strong>{stats["categories"]}</strong> categories<br>
            <strong>{stats["organic"]}</strong> organic items
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", width='stretch'):
        st.session_state.messages = []
        st.session_state.needs_response = False
        st.session_state.session_round = 0
        st.session_state.thread_id = str(uuid4())
        st.session_state.pop("pending_image", None)
        st.rerun()


st.markdown(
    """
    <div class="shop-hero">
        <div class="shop-kicker">Catalog agent</div>
        <h1>Find the right product faster.</h1>
        <p>
            Ask for budget, rating, organic, category, or image-based matches.
            The assistant checks the store catalog before it recommends or orders anything.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
metric_items = [
    ("Products", stats["products"]),
    ("Categories", stats["categories"]),
    ("Organic options", stats["organic"]),
]

for col, (label, value) in zip(metric_cols, metric_items):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">Quick requests</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Start with a common shopping constraint, then refine in chat.</div>',
    unsafe_allow_html=True,
)

quick_prompts = [
    "Find organic honey under $20 with at least 4.5 stars.",
    "Compare the best pasta options under $5.",
    "Recommend a high-rated pantry staple for meal prep.",
    "I need household basics under $10 with good reviews.",
]

quick_cols = st.columns(4)
for index, prompt in enumerate(quick_prompts):
    with quick_cols[index]:
        if st.button(prompt, width='stretch', disabled=st.session_state.needs_response):
            submit_prompt(prompt)

st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            Tell me what you are shopping for. Useful constraints include price ceiling,
            organic preference, rating threshold, category, and whether you are ready to buy.
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    render_message(msg)

if st.session_state.needs_response and st.session_state.messages:
    render_assistant_response()

if prompt := st.chat_input(
    "Ask for a product, comparison, or checkout...",
    disabled=st.session_state.needs_response,
):
    submit_prompt(prompt)
