import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

def format_docs(docs):
    """
    Formats a list of document objects into a single string, separating each document's content with two newlines.

    Args:
        docs (list): A list of document objects, each expected to have a 'page_content' attribute.

    Returns:
        str: A single string containing the concatenated 'page_content' of all documents, separated by two newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4.1-nano",
        temperature=0.2,
        )
    
    vector_store = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
    )
    
    template = """use the following peices of contet to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up answer.
    Use three sentences maximum and keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    
    Helpful Answer: 
    
    """
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"question": RunnablePassthrough(), "context": vector_store.as_retriever() | format_docs}
        | custom_rag_prompt
        | llm
    )
    
    query = "What is Pinecone?"
    
    rag_result = rag_chain.invoke(query)
    
    print(rag_result.content)