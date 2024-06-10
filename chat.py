import streamlit as st
import pandas as pd
import json
import os
from agent import query_agent
from langchain.agents import create_pandas_dataframe_agent
from google.cloud import aiplatform
from google.oauth2 import service_account
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def decode_response(response: str) -> dict:
    return json.loads(response, strict=False)

def write_response(response_dict: dict):
    if "answer" in response_dict:
        st.write(response_dict["answer"])
    elif any(chart_type in response_dict for chart_type in ["bar", "pie", "scatter", "line"]):
        chart_type = next(iter(response_dict))
        code = response_dict[chart_type]['python_code']
        exec(code, globals())
        st.plotly_chart(fig, theme=None, use_container_width=True)
    elif "manipulation" in response_dict:
        code = response_dict['manipulation']['python_code']
        exec(code, globals())
        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='df.csv',
            mime='text/csv', key="manipulation"+str(uniq))
        st.write(df)
    elif "table" in response_dict:
        code = "df_temp=" + response_dict['table']['python_code']
        exec(code, globals())
        csv = convert_df(df_temp)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='df.csv',
            mime='text/csv', key="table"+str(uniq))
        st.write(df_temp)
    else:
        st.write(response_dict)

st.title("👨‍💻 Chat with your data")
st.write("Please upload your CSV file below.")

with st.sidebar:
    option = st.selectbox("Model Temperature", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

data = st.file_uploader("Upload a CSV")
if data is not None:
    df = pd.read_csv(data)
    reset_df = df
    st.write(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

len_msg = len(st.session_state.messages)
uniq = len_msg

for message in st.session_state.messages:
    try:
        uniq += 1
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "{" in message["content"]:
                    decoded_history = decode_response(message["content"])
                else:
                    decoded_history = message["content"]
                if isinstance(decoded_history, dict):
                    write_response(decoded_history)
                else:
                    st.write(str(decoded_history))
            else:
                st.markdown(message["content"])
    except Exception as e:
        st.error(f"An error occurred: please try another query/question")

if prompt := st.chat_input("Enter your prompt:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    uniq += 1
    try:
        with st.spinner('Please wait...'):
            memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
            local_llm = 'llama3'
            llm = ChatOllama(model=local_llm, temperature=0)
            agent = create_pandas_dataframe_agent(llm, df, memory=memory, verbose=True)
            response = query_agent(agent=agent, query=prompt)
            if "{" in response:
                decoded_response = decode_response(response)
            else:
                decoded_response = response
            if isinstance(decoded_response, dict):
                decoded_response = decoded_response
            else:
                decoded_response = str(decoded_response)
        with st.chat_message("assistant"):
            st.write("Here's the answer:")
            write_response(decoded_response)
    except Exception as e:
        st.error(f"An error occurred: please try another query/question")
    uniq += 1
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat", key=reset_button_key)
    if reset_button:
        st.session_state.messages = []
