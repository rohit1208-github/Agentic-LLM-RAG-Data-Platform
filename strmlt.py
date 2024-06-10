import streamlit as st
import langchain_helper as lch

st.title("Query Student Records")

question = st.sidebar.text_area(label=f"Ask a question")

if st.sidebar.button("Get Answer"):
    response = lch.get_insights(question)

    st.write("Answer:", response)