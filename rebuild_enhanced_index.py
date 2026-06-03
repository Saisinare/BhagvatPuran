#!/usr/bin/env python3
"""
Rebuild the RAGita index with enhanced retrieval capabilities.
This script forces a rebuild of the FAISS index and metadata with improved text processing.
"""

import os
import shutil
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from RAG.pipeline.retriever import Retriever

def rebuild_index():
    """Force rebuild the index with enhanced text processing"""
    
    print("🔄 Rebuilding RAGita index with enhanced retrieval...")
    
    # Remove existing index files to force rebuild
    indices_dir = PROJECT_ROOT / "RAG" / "data" / "indices"
    
    if indices_dir.exists():
        print(f"📁 Removing existing index directory: {indices_dir}")
        shutil.rmtree(indices_dir)
    
    # Create new retriever - this will automatically rebuild the index
    print("🏗️  Building enhanced index with:")
    print("   ✓ Query expansion")
    print("   ✓ Conceptual keywords")
    print("   ✓ Improved text representation")
    print("   ✓ Multi-strategy retrieval")
    
    retriever = Retriever()
    
    print(f"✅ Enhanced index built successfully!")
    print(f"📊 Indexed {len(retriever.metas)} verses")
    
    # Test the enhanced retrieval with a sample query
    print("\n🧪 Testing enhanced retrieval...")
    test_queries = [
        "What is the Bhagavad Gita about?",
        "Who are the main characters?", 
        "What are the three gunas?",
        "What is the nature of the soul?",
        "How to find inner peace?"
    ]
    
    for query in test_queries:
        results = retriever.semantic_search(query, top_k=3)
        print(f"\nQuery: {query}")
        for i, result in enumerate(results, 1):
            verse_ref = f"{result['chapter']}:{result['verse']}"
            preview = result['english'][:100] + "..." if len(result['english']) > 100 else result['english']
            print(f"  {i}. [{verse_ref}] {preview}")
    
    print("\n🎉 Enhanced index rebuild complete!")
    print("🚀 Your RAGita system is now optimized for better retrieval performance!")

if __name__ == "__main__":
    rebuild_index()