import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template


custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_pdf_text(docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    """Split the extracted text into smaller chunks."""
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    """Create a vector store from the text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversationchain(vectorstore):
    """Set up a conversation chain with memory and Groq LLM."""
    groq_api = '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'
    llm = ChatGroq(temperature=0, model="llama3-8b-8192", api_key=groq_api)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True, 
                                      output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vectorstore.as_retriever(),
                            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                            memory=memory)
    return conversation_chain

def handle_question(question):
    """Handle user's question by passing it through the conversation chain."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF document first.")
        return


    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]


    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    """Main function to handle UI and interactions."""
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

   
    question = st.text_input("Ask a question from your document:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        
    
        if docs:
            if st.button("Process"):
                with st.spinner("Processing..."):
            
                    raw_text = get_pdf_text(docs)
                    if raw_text.strip() == "":
                        st.error("No valid text found in the PDFs. Please upload readable PDFs.")
                    else:
               
                        text_chunks = get_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversationchain(vectorstore)
                        st.success("Documents processed successfully. You can now ask questions.")
        else:
            st.warning("Please upload PDF documents to start.")

if __name__ == '__main__':
    main()
