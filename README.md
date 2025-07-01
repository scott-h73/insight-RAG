# WSE Insight RAG System

AI-powered Retrieval Augmented Generation (RAG) system for technical document queries. Built for Wave Swell Energy with enterprise-grade security and professional chat interface.

## 🚀 Features

- **Intelligent Document Search**: Natural language queries across technical documentation
- **Source Attribution**: Every answer includes document references for verification
- **Professional Security**: API key authentication, rate limiting, and audit logging
- **Beautiful Interface**: Modern chat interface with real-time responses
- **Multi-Format Support**: PDFs, Word docs, text files, and more
- **Enterprise Ready**: Scalable architecture with production deployment

## 📋 System Requirements

- Python 3.8+
- OpenAI API account
- Pinecone vector database account
- 2GB+ RAM recommended

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/wse-insight-rag.git
cd wse-insight-rag
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart python-dotenv llama-index llama-index-vector-stores-pinecone llama-index-llms-openai llama-index-embeddings-openai pinecone
```

### 3. Configure Environment
```bash
cp .env.template .env
# Edit .env with your actual API keys
```

### 4. Ingest Documents
```bash
# Add your documents to a 'documents' folder
python ingest_documents.py
```

### 5. Start Server
```bash
python rag_server.py
```

Visit `http://localhost:5000` and use PIN: `2025`

## 📁 File Structure

```
wse-insight-rag/
├── rag_server.py          # Main production server
├── index.html             # Chat interface
├── ingest_documents.py    # Document processing pipeline
├── query_documents.py     # Local testing utilities
├── rag_api.py            # Original development version
├── .env.template         # Environment setup template
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=llamaindex
VALID_API_KEYS=wse-demo-key-2025
ALLOWED_ORIGINS=https://*.replit.app,http://localhost:*
```

### Pinecone Setup

1. Create Pinecone account at https://pinecone.io
2. Create new index with:
   - **Dimensions**: 1024
   - **Metric**: cosine
   - **Pod Type**: Starter (free tier)

## 🚀 Deployment

### Local Development
```bash
python rag_server.py
# Access at http://localhost:5000
```

### Production (Replit)
1. Upload files to Replit
2. Set environment variables in Secrets
3. Run `python rag_server.py`

## 📖 Usage

### Adding Documents
1. Place documents in `documents/` folder
2. Run `python ingest_documents.py`
3. Choose whether to clear existing data
4. Documents are processed and indexed automatically

### Querying System
- **Web Interface**: Visit server URL, enter PIN, ask questions
- **Local Testing**: Use `python query_documents.py`
- **API Access**: POST to `/chat` endpoint with Bearer token

### Example Queries
- "What is the efficiency of the system?"
- "How does the UniWave200 turbine work?"
- "What are the key performance metrics?"

## 🔒 Security Features

- **Authentication**: PIN-based access + API key validation
- **Rate Limiting**: 100 requests per hour per IP
- **Input Validation**: Prevents malicious queries
- **Audit Logging**: All queries logged with timestamps
- **CORS Protection**: Configurable origin restrictions

## 🏗️ Architecture

```
User Query → FastAPI Server → LlamaIndex → Pinecone → OpenAI → Response
                ↓
           Authentication → Rate Limiting → Logging
```

**Components:**
- **FastAPI**: Web server and API framework
- **LlamaIndex**: RAG orchestration and query engine
- **Pinecone**: Vector database for document embeddings
- **OpenAI**: Language model (GPT-3.5-turbo) and embeddings

## 📊 Performance

- **Response Time**: < 3 seconds for most queries
- **Document Capacity**: Tested with 144 document chunks
- **Concurrent Users**: Supports multiple simultaneous queries
- **Accuracy**: High-quality responses with source attribution

## 🛠️ Development

### Adding New Features
1. Create feature branch: `git checkout -b feature-name`
2. Develop and test locally
3. Update documentation
4. Submit pull request

### Testing
```bash
# Test document ingestion
python ingest_documents.py

# Test query functionality
python query_documents.py

# Test server health
curl http://localhost:5000/health
```

## 📈 Business Applications

- **Technical Documentation**: Equipment manuals, specifications
- **Research Analysis**: Scientific papers, reports
- **Compliance**: Regulatory documents, standards
- **Knowledge Management**: Institutional knowledge preservation
- **Customer Support**: Technical troubleshooting guides

## 🔮 Roadmap

- [ ] Agentic RAG capabilities with external API integration
- [ ] Multi-language document support
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] Enterprise SSO integration
- [ ] Custom model fine-tuning

## 📄 License

Proprietary - Wave Swell Energy Ltd.

## 📞 Support

For technical support or business inquiries, contact:
- Email: support@waveswell.com
- Documentation: [Internal Wiki]
- Issues: Use GitHub Issues for bug reports

---

**Built with ❤️ for Wave Swell Energy**