# Medium Analyzer RAG

Built a simple RAG system to explore Pinecone's capabilities and understand how vector databases work in real applications. Took about two days to get running properly.

## Key learnings
- Successfully implemented RAG with Pinecone (after some trial and error)
- Gained hands-on experience with text embeddings
- Vector databases are more approachable than I initially thought
- LangChain is mind-blowing

## Implementation steps
1. Install dependencies: `pipenv install` (or use pip if you prefer)
2. Configure environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_key
   PINECONE_INDEX_NAME=your_index_name
   ```
3. Execute: `python main.py`