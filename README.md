# 🤖 Multi-Agent RAG System (MuRAG1)

An advanced **Agentic Retrieval-Augmented Generation** system that combines intelligent document processing, multimodal analysis, and adaptive learning capabilities using Google's Gemini AI.

## 🌟 Features

### 🧠 **Agentic Intelligence**
- **Smart Query Classification**: Automatically categorizes queries (summary, comparison, analysis, etc.)
- **Adaptive Planning**: Creates dynamic execution plans based on query type and context
- **Self-Reflection**: Evaluates and improves response quality automatically
- **Continuous Learning**: Adapts strategies based on performance history

### 📖 **Multimodal Document Processing**
- **PDF Processing**: Extract and index text from PDF documents
- **Image Analysis**: Analyze images using Gemini Vision API
- **OCR Capabilities**: Extract text from images automatically
- **Combined Analysis**: Seamlessly combine text and visual information

### 🛠️ **Advanced Tools**
- Document search and retrieval
- Intelligent summarization
- Comparative analysis
- In-depth content analysis
- Context enrichment
- Multimodal fusion

### 💡 **Key Capabilities**
- **Memory System**: Maintains conversation context and learning history
- **Performance Tracking**: Monitors and optimizes tool performance
- **Error Handling**: Robust retry mechanisms with exponential backoff
- **Connectivity Testing**: Built-in API connection diagnostics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google AI (Gemini) API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aladimassi/murag1.git
   cd murag1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   ```bash
   # PowerShell
   $env:GOOGLE_API_KEY="your_gemini_api_key_here"
   
   # Or create a .env file
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

4. **Run the system**
   ```bash
   python agentic_rag.py
   ```

## 📚 Usage

### Main System
The main agentic RAG system provides a comprehensive interface:

```bash
python agentic_rag.py
```

**Available Features:**
- 📄 Add PDF documents
- 🖼️ Add single images
- 📁 Batch process image directories
- 🤖 Multimodal chat interface
- 💾 Save/load knowledge bases
- 📊 Performance statistics
- 🛠️ Tool testing
- 🔌 API connectivity testing

### Communication Agent
For profile analysis and relationship building:

```bash
python run_communication_agent.py
```

### Simple API Server
For web-based access:

```bash
python simple_server.py
```

## 🏗️ Architecture

### Core Components

1. **AgenticRAG**: Main orchestrator with multimodal capabilities
2. **QueryClassifier**: Intelligent query categorization with learning
3. **AgentPlanner**: Dynamic plan creation and optimization
4. **ConversationMemory**: Context and performance tracking
5. **Tools**: Specialized processors for different task types

### Tool Ecosystem

- **DocumentSearchTool**: Vector-based document retrieval
- **SummarizationTool**: Intelligent content summarization
- **ComparisonTool**: Multi-document comparison analysis
- **AnalysisTool**: Deep content analysis
- **ImageAnalysisTool**: Gemini Vision-powered image understanding
- **OCRTool**: Text extraction from images
- **MultimodalAnalysisTool**: Combined text-image analysis
- **SelfReflectionTool**: Response quality evaluation

## 📦 Project Structure

```
murag1/
├── agentic_rag.py          # Main agentic RAG system
├── agent.py                # Communication agent for profiles
├── agent_executor.py       # MCP agent executor
├── requirements.txt        # Python dependencies
├── run_communication_agent.py  # Communication agent runner
├── simple_server.py        # FastAPI server
├── test_setup.py          # Setup verification
├── demo_agentic.py        # System demonstration
├── advanced_demo.py       # Advanced features demo
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Gemini AI API key (required)

### Customization
- Modify model parameters in the class constructors
- Adjust chunk sizes and overlap in text processing
- Configure retry logic and timeouts
- Customize classification keywords and patterns

## 📊 Performance Features

### Learning & Adaptation
- **Pattern Recognition**: Learns from successful query patterns
- **Strategy Optimization**: Adapts execution strategies based on results
- **Performance Metrics**: Tracks success rates and response quality
- **Continuous Improvement**: Refines approaches over time

### Monitoring
- Real-time performance statistics
- Tool effectiveness tracking
- Classification accuracy metrics
- API connectivity monitoring

## 🔗 API Integration

### Gemini AI APIs Used
- **Gemini 1.5 Flash**: Text generation and analysis
- **Gemini Vision**: Image analysis and OCR

### Supported Models
- Text: `gemini-1.5-flash`, `gemini-2.0-flash-exp`
- Vision: `gemini-1.5-flash` (multimodal)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] Advanced multimodal fusion techniques
- [ ] Integration with more AI models
- [ ] Enhanced learning algorithms
- [ ] Web-based user interface
- [ ] Distributed processing capabilities
- [ ] Plugin architecture for custom tools

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify your `GOOGLE_API_KEY` is set correctly
   - Check internet connectivity
   - Use the built-in connectivity test (option 9)

2. **Memory Issues**
   - Reduce batch sizes for large documents
   - Clear conversation memory periodically
   - Monitor system resources

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Getting Help
- Check the built-in help system (`help` command in chat)
- Review the troubleshooting tools (connectivity test, tool testing)
- Open an issue on GitHub for bugs or feature requests

## 👨‍💻 Author

Created by **Talan SummerCamp 25 Project**

---

*An intelligent, adaptive RAG system that learns and evolves with every interaction.*
