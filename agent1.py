import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="FloatChat", page_icon="🌊", layout="wide")
st.title("🌊 FloatChat: AI Interface for ARGO Data")
st.markdown("""
Welcome! Ask questions about ARGO oceanographic data.
- **Metadata questions**: "Which floats are available?"
- **Data queries**: "What is the max temperature for float 13857?"
- **Plots**: "Plot the temperature profile for float 13857"
""")

# --- Config ---
SQLITE_URI = "sqlite:///argo_data.db"
CHROMA_PERSIST_DIR = "./chroma_db"

# --- Setup Resources ---
@st.cache_resource
def setup_resources():
    db_engine = create_engine(SQLITE_URI)
    sql_database = SQLDatabase(engine=db_engine)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    return sql_database, vector_store, db_engine

try:
    sql_db, vector_db, engine = setup_resources()
except Exception as e:
    st.error(f"Failed to initialize resources: {e}")
    st.stop()

# --- LLMs ---
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
simple_llm = OpenAI(temperature=0)

# --- SQL Agent ---
sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=simple_llm)
sql_agent_executor = create_sql_agent(
    llm=simple_llm,
    toolkit=sql_toolkit,
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# --- Retrieval for metadata ---
retrieval_prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based only on the following context:
<context>
{context}
</context>
Question: {input}"""
)
document_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Helper Functions ---
def clean_sql_query(query: str) -> str:
    """Remove Markdown code fences or extra formatting from SQL."""
    return query.replace("```sql", "").replace("```", "").strip()

def format_sql_result(df: pd.DataFrame) -> str:
    """Return polished, human-readable summary of SQL results."""
    if df.empty:
        return "No results found."
    
    # Single-column results
    if len(df.columns) == 1:
        col = df.columns[0]
        values = df[col].tolist()
        if len(values) == 1:
            return f"The {col} is {values[0]:.2f}."
        else:
            values_str = ", ".join(f"{v:.2f}" for v in values)
            return f"The {col} values are: {values_str}."
    
    # Multiple columns
    return df.to_string(index=False)

def plot_temperature_profile(df: pd.DataFrame) -> str:
    """Plot temperature vs. pressure and return summary text."""
    if 'temperature' not in df.columns or 'pressure' not in df.columns or df.empty:
        return "Cannot plot: required columns not found or data is empty."

    df_sorted = df.sort_values('pressure', ascending=False)
    fig, ax = plt.subplots()
    ax.plot(df_sorted['temperature'], df_sorted['pressure'], marker='o')
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Pressure (dbar)")
    ax.invert_yaxis()
    ax.grid(True)
    st.pyplot(fig)

    return f"Plot generated. Temperature ranges from {df['temperature'].min():.2f} to {df['temperature'].max():.2f} °C, Pressure ranges from {df['pressure'].min():.2f} to {df['pressure'].max():.2f} dbar."

def process_sql_agent_output(agent_output: str) -> str:
    """Detect numeric output and format it directly; return None if not numeric."""
    agent_output = agent_output.strip()
    try:
        val = float(agent_output)
        return f"The result is {val:.2f}."
    except ValueError:
        return None

# --- Chat Session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the ARGO data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_content = ""
        with st.spinner("Processing..."):
            try:
                prompt_lower = prompt.lower()

                # --- Plotting requests ---
                if any(k in prompt_lower for k in ["plot", "chart", "visualize"]):
                    agent_prompt = f"""
                    User request: '{prompt}'.
                    Generate a SQL query to retrieve data from 'measurements' table
                    (columns: pressure, temperature, latitude, longitude, timestamp, float_id).
                    Only return raw SQL.
                    """
                    sql_response = sql_agent_executor.invoke({"input": agent_prompt})
                    sql_query = clean_sql_query(sql_response['output'])
                    st.code(sql_query, language='sql')

                    df = pd.read_sql(text(sql_query), engine)
                    response_content = plot_temperature_profile(df)

                # --- Metadata / float info ---
                elif any(k in prompt_lower for k in ["float", "which floats", "what floats"]):
                    response = retrieval_chain.invoke({"input": prompt})
                    response_content = response['answer']

                # --- Special case: Temperature at maximum pressure ---
                elif "temperature at maximum pressure" in prompt_lower:
                    sql_query = """
                        SELECT temperature 
                        FROM measurements
                        WHERE pressure = (SELECT MAX(pressure) FROM measurements);
                    """
                    df = pd.read_sql(text(sql_query), engine)
                    response_content = format_sql_result(df)

                # --- General SQL queries ---
                else:
                    sql_response = sql_agent_executor.invoke({"input": prompt})
                    agent_output = clean_sql_query(sql_response['output'])

                    # Detect numeric output
                    polished_result = process_sql_agent_output(agent_output)
                    if polished_result:
                        response_content = polished_result
                    else:
                        df = pd.read_sql(text(agent_output), engine)
                        response_content = format_sql_result(df)

            except Exception as e:
                response_content = f"Error: {e}"
                st.error(response_content)

        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.markdown(response_content)
