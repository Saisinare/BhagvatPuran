#!/usr/bin/env python3
"""
Interactive runner script for RAGita - Bhagavad Gita RAG System
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_queries():
    """Demo different types of queries to show versatility"""
    return [
        "How can I control my thoughts?",
        "What does the Gita say about success?",
        "I'm feeling stressed at work, any advice?",
        "What is dharma according to the Gita?",
        "How should I handle difficult people?",
        "What's the meaning of life according to Krishna?",
        "I'm struggling with motivation, help me!",
        "Tell me about Arjuna's dilemma",
        "How can I be more peaceful?",
        "What does the Gita teach about relationships?"
    ]

def get_user_query():
    """Get query from user input"""
    print("\n" + "="*60)
    print("💬 What would you like to know? Ask me anything!")
    print("   (Type 'quit', 'exit', or 'q' to stop)")
    print("   (Type 'examples' to see sample questions)")
    print("   (Type 'help' for more options)")
    print("="*60)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                print("❌ Please enter a question!")
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n🙏 Thank you for using RAGita! Om shanti.")
                return None
                
            if query.lower() == 'examples':
                show_examples()
                continue
                
            if query.lower() == 'help':
                show_help()
                continue
                
            return query
            
        except KeyboardInterrupt:
            print("\n\n🙏 Thank you for using RAGita! Om shanti.")
            return None
        except EOFError:
            print("\n\n🙏 Thank you for using RAGita! Om shanti.")
            return None

def show_examples():
    """Show example queries"""
    print("\n📚 Example Questions You Can Ask:")
    print("-" * 40)
    demo_queries_list = demo_queries()
    for i, demo_query in enumerate(demo_queries_list, 1):
        print(f"{i:2d}. {demo_query}")
    print("\n💡 Or ask about any topic - I'll help you through Gita's wisdom!")

def show_help():
    """Show help information"""
    print("\n🆘 RAGita Help:")
    print("-" * 20)
    print("• Ask me anything - advice, philosophy, questions about the Gita")
    print("• I automatically detect the type of your question")
    print("• I provide detailed, well-cited responses from the Bhagavad Gita")
    print("• Type 'examples' to see sample questions")
    print("• Type 'quit' or 'exit' to stop")
    print("• I can discuss any topic through Gita's wisdom!")

def process_query(query, persona="versatile"):
    """Process a single query and display results"""
    try:
        # Import the orchestrator function
        from RAG.pipeline.orchestrator import answer_query
        
        print(f"\n🔍 Processing: {query}")
        print("⏳ Searching through the Bhagavad Gita...")
        
        # Get the answer
        result = answer_query(query, persona=persona)
        
        # Show detected persona and query type
        if result.get('query_type') != 'manual':
            print(f"\n🎭 Detected: {result.get('query_type', 'unknown')} question")
            print(f"👤 Using: {result.get('persona', 'versatile')} approach")
        
        # Show the main answer
        print(f"\n💡 Answer:")
        print("-" * 30)
        print(result.get('answer_raw', 'No answer available'))
        
        # Show structured steps if available
        if result.get('steps'):
            print(f"\n📋 Practical Steps:")
            print("-" * 20)
            for i, step in enumerate(result.get('steps', []), 1):
                step_text = step.get('text', '') if isinstance(step, dict) else str(step)
                citation = step.get('citation', '') if isinstance(step, dict) else ''
                print(f"{i}. {step_text}")
                if citation:
                    print(f"   📖 [{citation}]")
        
        # Show statistics
        print(f"\n📊 Found {len(result.get('candidates', []))} relevant passages")
        
        if result.get('verification'):
            supported = result.get('supported_claims', 0)
            total = result.get('total_claims_checked', 0)
            if total > 0:
                print(f"✅ Verified {supported}/{total} claims ({supported/total*100:.1f}% supported)")
        
        # Show any warnings
        if result.get('rerank_warning'):
            print(f"\n⚠️  Warning: {result['rerank_warning']}")
        if result.get('verification_warning'):
            print(f"\n⚠️  Warning: {result['verification_warning']}")
        if result.get('parse_warnings'):
            print(f"\n⚠️  Parse warnings: {', '.join(result['parse_warnings'])}")
            
    except Exception as e:
        print(f"\n❌ Error processing your question: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main interactive function"""
    print("🕉️  RAGita - Your Versatile Bhagavad Gita Companion")
    print("=" * 60)
    print("Welcome! I can help you with any question using the wisdom of the Bhagavad Gita.")
    print("Ask me about life, philosophy, advice, or anything else!")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAGita - Interactive Bhagavad Gita Companion")
    parser.add_argument("--query", "-q", type=str, help="Ask a specific question directly")
    parser.add_argument("--persona", "-p", type=str, default="versatile", 
                       choices=["versatile", "scholarly", "devotional", "practical", "philosophical", "counselor", "teacher"],
                       help="Choose a specific persona (default: auto-detect)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode (default)")
    
    args = parser.parse_args()
    
    try:
        if args.query:
            # Single query mode
            process_query(args.query, args.persona)
        else:
            # Interactive mode
            while True:
                query = get_user_query()
                if query is None:
                    break
                process_query(query, args.persona)
                
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
