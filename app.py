import streamlit as st


from groq import Groq
import os

client = Groq(api_key=("GROQ_API_KEY"))
st.title("AI Chatbot")
user_input=st.text_input("Ask something:")
if st.button("Send"):
    if user_input:
        response=client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"user","content":user_input}
            ]
        )
        st.write("Response:")
        st.write(response.choices[0].message.content)
