from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings

def get_insights(question):
    loader = CSVLoader(r"C:\Users\prohi\Desktop\edplus\tabdat\Scholarship_Recipients_And_Dollars_By_College_Code__Beginning_2009.csv")
    documents = loader.load()

    embeddings = GPT4AllEmbeddings()

    chroma_db = Chroma.from_documents(
        documents, embeddings, persist_directory="./chroma_db"
    )
    chroma_db.persist()

    llm = Ollama(model="llama3")

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Given this context: {context}, please directly answer the question: {question}.",
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )
    print(chroma_db.as_retriever())
    result = qa_chain({"query": question})
    return result