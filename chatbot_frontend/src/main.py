import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/hospital-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the customers, their mortgages,
        payment due dates, payments and fees in  a dummy banking system.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("""-Find all customers living in California.?""")
    st.markdown("""- What is the current wait time at wallace-hamilton branch?""")
    st.markdown(
        """- What is the email address of the customer with customer ID C001??""" )
    st.markdown("- What is the average duration in days for closed emergency visits?")
    st.markdown(
        """- What are the terms and conditions for the new mortgage product?"""
    )
    st.markdown("- What was the total late fee charged for customer Bob?")
    st.markdown("- What is the average billing amount for medicaid visits?")
    st.markdown("- Which physician has the lowest average visit duration in days?")
    st.markdown("- How much was billed for patient 789's stay?")
    st.markdown(
        """- How many active 'Adjustable-Rate' loans are held by customers 
        in New York?""")
    # st.markdown("- What is the next payment due date for customer Alice for her fixed rate mortgage?")
    # st.markdown(
    #     """- How many reviews have been written from
    #             patients in Florida?"""
    # )
    # st.markdown(
    #     """- For visits that are not missing chief complaints,
    #    what percentage have reviews?"""
    # )
    # st.markdown(
    #     """- What is the percentage of visits that have reviews for
    #     each hospital?"""
    # )
    # st.markdown(
    #     """- Which physician has received the most reviews for this visits
    #     they've attended?"""
    # )
    # st.markdown("- What is the ID for physician James Cooper?")
    # st.markdown(
    #     """- List every review for visits treated by physician 270.
    #     Don't leave any out."""
    # )


st.title("Banking System Chatbot")
st.info(
    """Ask me questions about product, promotions, bank branches, transactions, fees, payments,
    ,FAQs, and appointment times!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            This usually means the chatbot failed at generating a query to
            answer your question. Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
