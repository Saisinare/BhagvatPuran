"""
Sample test data for RAGita evaluation.
Contains diverse queries with ground truth annotations.
"""

# Sample test dataset with ground truth annotations
SAMPLE_TEST_DATA = [
    {
        "query": "What is the Bhagavad Gita about? How many verses are there?",
        "ground_truth": {
            "relevant_docs": ["1:1", "18:78", "2:11", "2:12"],  # Key introductory verses
            "gold_answer": "The Bhagavad Gita is a 700-verse Hindu scripture that presents a philosophical dialogue between Prince Arjuna and Lord Krishna on the battlefield of Kurukshetra. It addresses fundamental questions about duty, righteousness, and the nature of reality through 18 chapters containing exactly 700 verses.",
            "expected_citations": ["1:1", "18:78"]
        }
    },
    {
        "query": "How can I control my thoughts and find inner peace?",
        "ground_truth": {
            "relevant_docs": ["2:62", "2:63", "6:5", "6:6", "6:19", "6:35"],
            "gold_answer": "The Gita teaches that controlling thoughts requires disciplined practice, self-control, and steady meditation. One must withdraw the mind from sense objects and focus it on the Self through persistent practice and detachment.",
            "expected_citations": ["2:62", "6:35", "6:19"]
        }
    },
    {
        "query": "What does Krishna say about action without attachment?",
        "ground_truth": {
            "relevant_docs": ["2:47", "3:19", "4:20", "18:9"],
            "gold_answer": "Krishna teaches that one should perform their prescribed duties without attachment to results. This is the path of karma yoga - acting for the sake of duty and service rather than personal gain.",
            "expected_citations": ["2:47", "3:19", "4:20"]
        }
    },
    {
        "query": "How should I deal with difficult people in my life?",
        "ground_truth": {
            "relevant_docs": ["12:13", "12:14", "16:1", "16:2", "16:3"],
            "gold_answer": "The Gita advocates responding to difficult people with compassion, patience, and understanding. One should maintain equanimity, avoid hatred, and practice forgiveness while protecting one's own dharma.",
            "expected_citations": ["12:13", "12:14", "16:1"]
        }
    },
    {
        "query": "What is dharma according to the Gita?",
        "ground_truth": {
            "relevant_docs": ["3:35", "18:47", "2:31", "4:7", "4:8"],
            "gold_answer": "Dharma in the Gita refers to righteous duty, moral law, and natural order. It includes one's duty according to their nature and circumstances, maintaining cosmic and social harmony, and upholding righteousness.",
            "expected_citations": ["3:35", "18:47", "2:31"]
        }
    }
]

# Additional test queries for specific aspects
RETRIEVAL_TEST_QUERIES = [
    {
        "query": "Tell me about Arjuna's dilemma at the beginning",
        "ground_truth": {
            "relevant_docs": ["1:28", "1:29", "1:30", "1:46", "2:1", "2:2"],
        }
    },
    {
        "query": "What are the three gunas mentioned in the Gita?",
        "ground_truth": {
            "relevant_docs": ["14:5", "14:6", "14:7", "14:8", "18:40"],
        }
    }
]

# Grounding-focused test queries (for testing citation quality)
GROUNDING_TEST_QUERIES = [
    {
        "query": "Quote Krishna's exact words about performing duty without attachment",
        "ground_truth": {
            "relevant_docs": ["2:47", "3:19"],
            "requires_exact_citation": True
        }
    }
]

def get_full_test_dataset():
    """Get the complete test dataset combining all query types."""
    return SAMPLE_TEST_DATA + RETRIEVAL_TEST_QUERIES + GROUNDING_TEST_QUERIES

def get_minimal_test_dataset():
    """Get a minimal dataset for quick testing."""
    return SAMPLE_TEST_DATA[:3]