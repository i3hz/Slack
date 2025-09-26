import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

# LangChain v0.2+ Updates: Modernized imports and chains
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
# --- Main Application Logic ---
# SET PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="FloatChat", page_icon="🌊", layout="wide")
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
SQLITE_URI = "sqlite:///argo_data.db"
CHROMA_PERSIST_DIR = "./chroma_db"
# You will need an OpenAI API key set as an environment variable: OPENAI_API_KEY

# --- Database and Vector Store Setup (Cached for performance) ---
@st.cache_resource
def setup_resources():
    """Connects to the database and loads the vector store."""
    print("Setting up resources...")
    # Setup SQL Database
    db_engine = create_engine(SQLITE_URI)
    sql_database = SQLDatabase(engine=db_engine)

    # Setup Vector Store using OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    print("Resources are set up.")
    return sql_database, vector_store, db_engine

st.title("🌊 FloatChat: AI Interface for ARGO Data")

st.markdown("""
Welcome to FloatChat! Ask questions in natural language to explore and visualize ARGO ocean data.
- **For data discovery**, ask about available floats (e.g., "Which floats are in the database?")
- **For data queries**, ask for specific information (e.g., "What is the max temperature for float 13857?")
- **For visualizations**, ask for plots (e.g., "Plot the temperature profile for float 13857")
""")

# Initialize resources
try:
    sql_db, vector_db, engine = setup_resources()
except Exception as e:
    st.error(f"Failed to initialize resources. Please ensure your `argo_data.db` and `chroma_db` directories are present. Error: {e}")
    st.stop()

# --- Agent and Tool Creation (Using Modern LangChain) ---
# Use a Chat Model, which is standard for new LangChain applications
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
def clean_sql_query(query: str) -> str:
    """Remove Markdown code fences or any extra formatting from SQL."""
    if not query:
        return query
    # Remove ```sql or ``` fences
    query = query.replace("```sql", "").replace("```", "").strip()
    return query


# Tool 1: Modern LCEL-based Retrieval Chain for metadata questions
retrieval_prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based only on the following context:
    <context>
    {context}
    </context>
    Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retriever = vector_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Tool 2: SQL Agent for data queries
sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=OpenAI(temperature=0)) # SQL Agent often works well with simpler completion models
sql_agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=sql_toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# --- User Interaction ---
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
        with st.spinner("Thinking..."):
            prompt_lower = prompt.lower()
            
            # Route to the appropriate tool based on keywords
            if "plot" in prompt_lower or "chart" in prompt_lower or "visualize" in prompt_lower:
                st.write("Generating plot data...")
                try:
                    # Ask the agent to generate a SQL query to get the data
                    agent_prompt = f"""
                    Given the user's request: '{prompt}', generate a SQL query to retrieve the necessary data from the 'measurements' table.
                    The table has columns: 'pressure', 'temperature', 'latitude', 'longitude', 'timestamp', 'float_id'.
                    For a profile plot, you will need pressure and temperature for a specific float.
                    Only return the raw SQL query.
                    """
                    query_response = sql_agent_executor.invoke({"input": agent_prompt})
                    sql_query = query_response['output']
                    
                    st.code(sql_query, language='sql')

                    # Execute the query and plot
                    with engine.connect() as conn:
                        df = pd.read_sql(text(sql_query), conn)

                    if not df.empty and 'temperature' in df.columns and 'pressure' in df.columns:
                        # For oceanographic plots, it's conventional to have pressure (depth) decreasing on the y-axis
                        df = df.sort_values('pressure', ascending=False)
                        st.line_chart(df.rename(columns={'pressure': 'Pressure (dbar)', 'temperature': 'Temperature (C)'}).set_index('Temperature (C)')['Pressure (dbar)'])
                        response_content = "Here is the plot you requested."
                    else:
                        response_content = "I couldn't retrieve the necessary data to create a plot from that query."
                        st.warning(response_content)

                except Exception as e:
                    response_content = f"Could not generate plot. Error: {e}"
                    st.error(response_content)

            elif "float" in prompt_lower or "what floats" in prompt_lower or "which floats" in prompt_lower:
                st.write("Searching metadata...")
                response = retrieval_chain.invoke({"input": prompt})
                response_content = response['answer']
                st.write(response_content)
            else:
                st.write("Querying SQL database...")
                try:
                    response = sql_agent_executor.invoke({"input": prompt})
                    response_content = response['output']
                    st.write(response_content)
                except Exception as e:
                    response_content = f"An error occurred: {e}"
                    st.error(response_content)
                    
        st.session_state.messages.append({"role": "assistant", "content": response_content})

