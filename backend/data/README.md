# Knowledge Base Data Folder

This folder contains the knowledge base files that will be used to create the AI assistant's vector database.

## Supported File Types

- **PDF files** (`.pdf`) - Startup guides, business books, research papers
- **Text files** (`.txt`) - Notes, guides, documentation

## How to Use

1. **Add your knowledge files** to this folder:
   - Copy PDF files (startup guides, business books, etc.)
   - Add text files with your notes and knowledge
   - The more relevant content, the better the AI responses

2. **Run the vector store creation script**:
   ```bash
   cd backend/vector_store_startup
   python create_vectorstore.py
   ```

3. **The script will**:
   - Read all PDF and text files in this folder
   - Extract text content from each file
   - Create embeddings for the content
   - Build a FAISS vector database for fast similarity search

## Example Files You Can Add

- Startup guides and business books (PDF)
- Market research reports (PDF)
- Business model templates (PDF/TXT)
- Financial planning guides (PDF/TXT)
- Legal and compliance documents (PDF/TXT)
- Marketing and sales strategies (PDF/TXT)
- Technology stack guides (PDF/TXT)
- Case studies and success stories (PDF/TXT)

## File Naming

Use descriptive names for your files:
- `startup_validation_guide.pdf`
- `business_model_canvas.txt`
- `funding_strategies.pdf`
- `team_building_guide.txt`

## Current Files

- `startup_guide.txt` - Sample startup knowledge base (can be replaced with your own content)

## Notes

- The script will automatically detect and process all PDF and text files
- If no files are found, it will use a sample knowledge base
- Large files may take longer to process
- Make sure PDFs are text-based (not scanned images) for best results 