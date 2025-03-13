import os
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import PyPDF2
import streamlit as st

load_dotenv(override=True)

llm= ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_file):
    pdf_file_reader=PyPDF2.PdfReader(pdf_file)
    text=" "
    for page in pdf_file_reader.pages:
        text += page.extract_text() + "\n"
    return text

def generate_quiz_from_pdf(pdf_file):
    text=extract_text_from_pdf(pdf_file)
    quiz_prompt=PromptTemplate(input_variables=[text],
                               template="Create a quiz on the {text} that should include 5 Multiple Choice Questions,5 true or false questiona and 5 fill in the blanks")
    quiz_chain=LLMChain(llm=llm,prompt=quiz_prompt)
    return quiz_chain.run({"text": text})

def generate_summary_from_pdf(pdf_file):
    text=extract_text_from_pdf(pdf_file)
    summary_prompt=PromptTemplate(input_variables=[text],
                                  template="Write a detailed summary of the following text: {text}.,"
                                  "The summary should include all key points and provide a comprehensive overview of the topic. Ensure the summary is at least 200 words.")
    summary_chain=LLMChain(llm=llm,prompt=summary_prompt)
    return summary_chain.run({"text": text})

def generate_flashcards_from_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    flashcard_prompt = PromptTemplate(input_variables=["text"],
                                      template="Create flashcards from the following text: {text}. Each flashcard should have a question on one side and the answer on the other side.")
    flashcard_chain = LLMChain(llm=llm, prompt=flashcard_prompt)
    return flashcard_chain.run({"text": text})


st.title("PDF to Study Material Generator")
st.write("This app uses AI to generate study material from PDF files. You can create quiz, summaries, and flashcards from the text in the PDF.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if st.button("Generate Quiz"):
    quiz=generate_quiz_from_pdf(uploaded_file)
    st.subheader("Generated Quiz")
    st.write(quiz)

if st.button("Generate Summary"):
    summary = generate_summary_from_pdf(uploaded_file)
    st.subheader("Generated Summary")
    st.write(summary)

if st.button("Generate Flashcards"):
    flashcards = generate_flashcards_from_pdf(uploaded_file)
    st.subheader("Generated Flashcards")
    st.write(flashcards)
