# Import things that are needed generically
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from fastapi import FastAPI
from langchain.vectorstores import Chroma
import chromadb




llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
documents_directory: str = "/Users/hanif/Desktop/LLM_Agent/data/palestine_aljazeera"
collection_name: str = "palestine"
persist_directory: str = "/Users/hanif/Desktop/LLM_Agent/data/database/chroma_db"

def create_agent():

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    

    langchain_chroma = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    retriever = langchain_chroma.as_retriever()
    palestine_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    db = SQLDatabase.from_uri("sqlite:////Users/hanif/Desktop/LLM_Agent/data/database/house.db", sample_rows_in_table_info=4)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)


    tools = [
        Tool(
            name="Palestine, Gaza and Israel related QA System",
            func=palestine_qa.run,
            description="useful for when you need to answer questions about platestine and gaza war related issue. Input should be a fully formed question.",
        ),
        Tool(
            name="SQL qa system",
            func=db_chain.run,
            description="useful for when you need to answer questions about house. Input should be a fully formed question.",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    )
    return agent

def main():
    # We use a simple input loop.
    agent = create_agent()
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")
        agent_response = agent.run(query)
        print(agent_response)

main()


"""

# initiating fastapi instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get("/")
def home():
    return {"message": "Talk to your LLM-agent"}

# Defining path operation for prediction endpoint
@app.post("/agent", description='enter your query on house price or gaza-palestine issue')
def agent(text: str):
    agent_message = create_agent().run(text)
    return  agent_message

    """