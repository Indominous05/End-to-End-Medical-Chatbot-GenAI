# End-to-End-Medical-Chatbot-GenAI
🩺 Medical AI Chatbot (RAG-based)

An intelligent Retrieval-Augmented Generation (RAG) powered medical chatbot that provides contextual, accurate responses using domain-specific medical data.

🚀 Features:
🔍 RAG Pipeline – Retrieves relevant medical context before generating answers
🧠 LLM Integration – Uses Gemini for natural language understanding
📄 PDF Knowledge Base – Extracts and processes medical documents
🧩 Vector Search (Pinecone) – Efficient similarity search using embeddings
⚡ Fast Retrieval – Top-k document retrieval for accurate responses
🌐 Interactive UI – Clean chatbot interface using HTML, CSS, JS
🔄 End-to-End Pipeline – From data ingestion → embeddings → retrieval → response


🏗️ Tech Stack:
Frontend: HTML, CSS, JavaScript
Backend: Flask (Python)
LLM: Gemini (Google Generative AI)
Embeddings: HuggingFace Sentence Transformers
Vector DB: Pinecone
Frameworks: LangChain


🧠 Architecture Overview
User Query->
Retriever (Pinecone)->
Top-K Relevant Chunks->
LLM (Gemini)->
Final Answer
