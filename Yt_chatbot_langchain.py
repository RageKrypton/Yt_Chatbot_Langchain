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

def extract_video_id(url_or_id):
    """Extract YouTube video ID from URL or return ID if already provided"""
    import re
    
    if len(url_or_id) == 11 and url_or_id.replace('-', '').replace('_', '').isalnum():
        return url_or_id
    
    # YouTube URL patterns
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None

def is_valid_youtube_id(video_id):
    """Validate YouTube video ID format"""
    return video_id and len(video_id) == 11 and video_id.replace('-', '').replace('_', '').isalnum()

st.title("YouTube Chatbot using Langchain")
st.write("Ask questions about any YouTube video (with captions). Just paste the YouTube URL or enter the video ID below!")

video_input = st.text_input(
    "Enter YouTube URL or Video ID:",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo or just Gfr50f6ZBvo",
    help="You can paste the full YouTube URL or just the 11-character video ID"
)

video_id = None
if video_input:
    video_id = extract_video_id(video_input.strip())
    if not video_id:
        st.error("Invalid YouTube URL or video ID. Please check your input.")
        st.info("**Supported formats:**\n- https://www.youtube.com/watch?v=VIDEO_ID\n- https://youtu.be/VIDEO_ID\n- VIDEO_ID (11 characters)")
    else:
        st.success(f"Valid video ID extracted: **{video_id}**")

if 'main_chain' not in st.session_state:
    st.session_state.main_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None

if st.button("Load Video Transcript"):
    if not video_input:
        st.error("Please enter a YouTube URL or video ID")
    elif not video_id:
        st.error("Please enter a valid YouTube URL or video ID")
    else:
        with st.spinner("Loading transcript..."):
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
                st.session_state.current_video_id = video_id

                st.session_state.chat_history = []
                st.success("Transcript loaded successfully! You can now chat below.")
                
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            except TranscriptsDisabled:
                st.error("This video has no captions available.")
            except Exception as e:
                st.error(f"Error fetching transcript: {e}")

if st.session_state.main_chain:
    st.markdown("---")
    st.subheader(f"Chatting about video: {st.session_state.current_video_id}")
    
    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.write(message)
    
    if prompt := st.chat_input("Ask a question about the video:"):
        st.session_state.chat_history.append(("You", prompt))
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.main_chain.invoke(prompt)
                    st.write(response)
                    st.session_state.chat_history.append(("Bot", response))
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("Bot", error_msg))

if st.session_state.chat_history:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Chat History ({len(st.session_state.chat_history)//2} messages)**")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

st.markdown("---")
st.markdown("**Note:** This chatbot can only answer questions based on the video's transcript. Make sure the video has captions enabled.")