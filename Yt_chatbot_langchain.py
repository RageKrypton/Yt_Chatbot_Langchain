import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("🎬 YouTube Chatbot using Langchain")
st.write("Ask questions about any YouTube video (with captions). Just enter the video ID below!")

video_id = st.text_input("Enter YouTube Video ID (e.g., Gfr50f6ZBvo):")

if 'main_chain' not in st.session_state:
    st.session_state.main_chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.button("Load Video Transcript"):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list).strip()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Transcript Context:
-------------------
{context}

Question:
---------
{question}
""",
            input_variables=['context', 'question']
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        st.session_state.main_chain = main_chain
        st.success("Transcript loaded successfully! You can now chat below.")

    except TranscriptsDisabled:
        st.error("This video has no captions available.")
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")

if st.session_state.main_chain:
    user_input = st.text_input("Ask a question about the video:")
    if st.button("Send"):
        if user_input:
            response = st.session_state.main_chain.invoke(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

if st.session_state.chat_history:
    st.subheader("Chat History:")
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")