from langchain_community.llms import Ollama
import streamlit as st 
import pandas as pd
from pandasai import SmartDataframe

def chat_with_csv(df,query):

    llm = Ollama(model="mistral")
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(query)
    return result

st.set_page_config(layout='wide')

st.title("Ask questions on ASU Tabular database")

input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

if input_csvs:
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(3),use_container_width=True)
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    if input_text:
        if st.button("Chat with csv"):
            st.info("Your Query: "+ input_text)
            result = chat_with_csv(data,input_text)
            st.success(result)




     