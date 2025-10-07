import streamlit as st
from food import reply

st.markdown(
    """
     <h1 style='text-align: left; color: blue; font-size: 20px; margin: 0;'>FOOD SAFETY BOT</h1>
    """,
    unsafe_allow_html=True
    )

with st.chat_message("assistant"):
    st.write("How can I help you today")
prompt = st.chat_input("Say something")
if prompt:
    st.chat_message("user").markdown(prompt)
    response = reply(prompt)
    st.chat_message("assistant").markdown(response)
    