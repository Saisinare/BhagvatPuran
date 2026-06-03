"""
Comprehensive test dataset for RAGita evaluation.
Contains diverse queries across multiple categories for robust evaluation.
"""

# Comprehensive test dataset with 25+ queries across diverse categories
COMPREHENSIVE_TEST_DATA = [
    # === FACTUAL QUESTIONS ===
    {
        "query": "What is the Bhagavad Gita about? How many verses are there?",
        "category": "factual",
        "ground_truth": {
            "relevant_docs": ["1:1", "18:78", "2:11", "2:12"],
            "gold_answer": "The Bhagavad Gita is a 700-verse Hindu scripture presenting philosophical dialogue between Arjuna and Krishna on duty, righteousness, and spiritual realization.",
        }
    },
    {
        "query": "Who are the main characters in the Bhagavad Gita?",
        "category": "factual", 
        "ground_truth": {
            "relevant_docs": ["1:1", "1:15", "1:20", "2:1"],
            "gold_answer": "The main characters are Prince Arjuna, Lord Krishna (his charioteer and guide), and Sanjaya who narrates the dialogue to King Dhritarashtra.",
        }
    },
    {
        "query": "What are the three gunas mentioned in the Gita?",
        "category": "factual",
        "ground_truth": {
            "relevant_docs": ["14:5", "14:6", "14:7", "14:8", "18:40"],
            "gold_answer": "The three gunas are sattva (goodness), rajas (passion), and tamas (ignorance), which are the fundamental qualities of nature.",
        }
    },
    {
        "query": "What is the significance of the battlefield of Kurukshetra?",
        "category": "factual",
        "ground_truth": {
            "relevant_docs": ["1:1", "1:2", "2:4", "2:5"],
            "gold_answer": "Kurukshetra is the sacred battlefield where the dialogue takes place, representing the field of righteousness and moral conflict.",
        }
    },
    
    # === PHILOSOPHICAL QUESTIONS ===
    {
        "query": "What is the nature of the soul according to Krishna?",
        "category": "philosophical",
        "ground_truth": {
            "relevant_docs": ["2:11", "2:12", "2:13", "2:20", "2:23"],
            "gold_answer": "The soul is eternal, indestructible, unborn, and unchanging. It transmigrates from body to body but is never actually born or dies.",
        }
    },
    {
        "query": "What does Krishna teach about the nature of reality?",
        "category": "philosophical", 
        "ground_truth": {
            "relevant_docs": ["7:4", "7:5", "9:4", "9:5", "15:7"],
            "gold_answer": "Reality consists of both material nature (prakriti) and spiritual nature (purusha), with Krishna as the ultimate source of both.",
        }
    },
    {
        "query": "What is the relationship between action and inaction?",
        "category": "philosophical",
        "ground_truth": {
            "relevant_docs": ["4:16", "4:17", "4:18", "3:4", "3:8"],
            "gold_answer": "True action is performed without attachment, while inaction in duty leads to degradation. The wise see inaction in action and action in inaction.",
        }
    },
    {
        "query": "What is Maya according to the Bhagavad Gita?",
        "category": "philosophical",
        "ground_truth": {
            "relevant_docs": ["7:14", "7:25", "18:61"],
            "gold_answer": "Maya is the divine illusory power that conceals the true nature of reality and causes beings to identify with the material world.",
        }
    },
    
    # === PRACTICAL GUIDANCE ===
    {
        "query": "How can I control my thoughts and find inner peace?",
        "category": "practical",
        "ground_truth": {
            "relevant_docs": ["2:62", "2:63", "6:5", "6:6", "6:19", "6:35"],
            "gold_answer": "Control thoughts through disciplined practice, meditation, detachment from sense objects, and focusing the mind on the divine.",
        }
    },
    {
        "query": "How should I deal with difficult people in my life?",
        "category": "practical",
        "ground_truth": {
            "relevant_docs": ["12:13", "12:14", "16:1", "16:2", "16:3"],
            "gold_answer": "Practice compassion, patience, forgiveness, and maintain equanimity while protecting your own dharma and well-being.",
        }
    },
    {
        "query": "What is the best way to handle failure and success?",
        "category": "practical",
        "ground_truth": {
            "relevant_docs": ["2:47", "2:48", "2:57", "12:17"],
            "gold_answer": "Remain equanimous in both success and failure, focusing on duty rather than results, and maintaining balance in all circumstances.",
        }
    },
    {
        "query": "How can I overcome fear and anxiety?",
        "category": "practical",
        "ground_truth": {
            "relevant_docs": ["4:10", "11:50", "18:58", "7:1"],
            "gold_answer": "Overcome fear through surrender to the divine, understanding the eternal nature of the soul, and cultivating unwavering faith.",
        }
    },
    {
        "query": "How should I make important life decisions?",
        "category": "practical",
        "ground_truth": {
            "relevant_docs": ["18:31", "18:32", "3:35", "18:47"],
            "gold_answer": "Make decisions using discriminative intelligence, following your dharma, consulting scriptures and wise teachers, and acting without attachment.",
        }
    },
    
    # === SPIRITUAL GUIDANCE ===
    {
        "query": "What are the different paths to spiritual realization?",
        "category": "spiritual",
        "ground_truth": {
            "relevant_docs": ["3:3", "4:11", "7:16", "12:8", "12:9"],
            "gold_answer": "The main paths are karma yoga (action), jnana yoga (knowledge), bhakti yoga (devotion), and dhyana yoga (meditation).",
        }
    },
    {
        "query": "How can I develop devotion to God?",
        "category": "spiritual",
        "ground_truth": {
            "relevant_docs": ["9:26", "9:27", "12:8", "12:10", "18:65"],
            "gold_answer": "Develop devotion through constant remembrance, offering all actions to God, regular worship, and surrendering the ego.",
        }
    },
    {
        "query": "What is the goal of meditation according to Krishna?",
        "category": "spiritual",
        "ground_truth": {
            "relevant_docs": ["6:15", "6:20", "6:21", "6:27", "8:12"],
            "gold_answer": "The goal is to achieve union with the divine, experience supreme peace, and realize one's true nature beyond material identification.",
        }
    },
    {
        "query": "How can I surrender to God completely?",
        "category": "spiritual",
        "ground_truth": {
            "relevant_docs": ["18:66", "9:34", "11:55", "12:6", "12:7"],
            "gold_answer": "Surrender by offering all actions to God, abandoning ego and desires, maintaining constant devotion, and trusting in divine grace.",
        }
    },
    
    # === ETHICAL QUESTIONS ===
    {
        "query": "What does Krishna say about duty and righteousness?",
        "category": "ethical",
        "ground_truth": {
            "relevant_docs": ["2:31", "3:35", "18:47", "4:7", "4:8"],
            "gold_answer": "One should perform their prescribed duty according to their nature and position, upholding righteousness and protecting dharma.",
        }
    },
    {
        "query": "How should I handle moral dilemmas?",
        "category": "ethical",
        "ground_truth": {
            "relevant_docs": ["2:7", "3:2", "18:31", "18:32"],
            "gold_answer": "Seek guidance from scriptures and wise teachers, use discriminative intelligence, and choose actions that uphold universal principles.",
        }
    },
    {
        "query": "What is the difference between right and wrong action?",
        "category": "ethical",
        "ground_truth": {
            "relevant_docs": ["16:23", "16:24", "17:11", "17:12", "18:23"],
            "gold_answer": "Right action follows scriptural guidelines, promotes universal welfare, and is performed with proper attitude and knowledge.",
        }
    },
    
    # === COMPLEX CONCEPTUAL QUESTIONS ===
    {
        "query": "Explain the concept of karma and its relationship to free will",
        "category": "complex",
        "ground_truth": {
            "relevant_docs": ["4:14", "4:17", "8:3", "9:9", "15:8"],
            "gold_answer": "Karma is the law of cause and effect in action. While bound by past karma, beings have free will to choose current actions and shape future destiny.",
        }
    },
    {
        "query": "What is the relationship between individual consciousness and universal consciousness?",
        "category": "complex",
        "ground_truth": {
            "relevant_docs": ["13:2", "15:7", "7:5", "10:20"],
            "gold_answer": "Individual consciousness is a fragment of universal consciousness, temporarily embodied but eternally connected to the supreme source.",
        }
    },
    {
        "query": "How does Krishna explain the paradox of action in inaction?",
        "category": "complex",
        "ground_truth": {
            "relevant_docs": ["4:16", "4:18", "4:19", "4:20"],
            "gold_answer": "The wise see inaction in action when acting without ego or attachment, and action in inaction when remaining inactive in duty.",
        }
    },
    
    # === COMPARATIVE QUESTIONS ===
    {
        "query": "What are the differences between the wise and the ignorant person?",
        "category": "comparative", 
        "ground_truth": {
            "relevant_docs": ["2:54", "2:55", "2:56", "5:20", "14:22"],
            "gold_answer": "The wise remain equanimous in pleasure and pain, act without attachment, and see the divine in all, while the ignorant are driven by desires and reactions.",
        }
    },
    {
        "query": "Compare the path of knowledge with the path of devotion",
        "category": "comparative",
        "ground_truth": {
            "relevant_docs": ["12:1", "12:5", "12:8", "12:12"],
            "gold_answer": "Both paths lead to the same goal; knowledge suits those capable of meditation on the formless, while devotion is easier for most people through worship of the personal form.",
        }
    },
    
    # === APPLIED WISDOM QUESTIONS ===
    {
        "query": "How can the teachings of the Gita help in modern workplace challenges?",
        "category": "applied",
        "ground_truth": {
            "relevant_docs": ["2:47", "3:19", "18:45", "18:46"],
            "gold_answer": "Apply principles of dutiful action without attachment to results, maintaining integrity, serving others, and seeing work as worship.",
        }
    },
    {
        "query": "What guidance does Krishna offer for maintaining relationships?",
        "category": "applied",
        "ground_truth": {
            "relevant_docs": ["12:13", "12:14", "12:15", "16:1", "16:2"],
            "gold_answer": "Practice compassion, patience, forgiveness, truthfulness, and see the divine in all beings while maintaining appropriate boundaries.",
        }
    }
]

# Additional challenging questions for advanced evaluation
ADVANCED_TEST_DATA = [
    {
        "query": "Reconcile Krishna's teaching on non-violence with his advice to Arjuna to fight",
        "category": "advanced",
        "ground_truth": {
            "relevant_docs": ["2:31", "2:32", "4:7", "4:8", "11:32"],
            "gold_answer": "Violence in protection of dharma and righteousness, performed as duty without personal hatred, differs from violence driven by selfish desires.",
        }
    },
    {
        "query": "How does the Gita address the problem of evil and suffering in the world?",
        "category": "advanced",
        "ground_truth": {
            "relevant_docs": ["4:7", "4:8", "7:12", "16:19", "16:20"],
            "gold_answer": "Evil arises from ignorance and imbalance of gunas. Divine incarnates periodically to restore dharma and guide beings toward righteousness.",
        }
    },
    {
        "query": "Explain the ultimate teaching of the Gita in one comprehensive statement",
        "category": "synthesis",
        "ground_truth": {
            "relevant_docs": ["18:65", "18:66", "9:34", "11:55"],
            "gold_answer": "Surrender completely to Krishna, perform all duties as offerings to the divine, and realize your eternal relationship with the Supreme through love and devotion.",
        }
    }
]

def get_comprehensive_test_dataset():
    """Get the full comprehensive test dataset."""
    return COMPREHENSIVE_TEST_DATA + ADVANCED_TEST_DATA

def get_category_specific_dataset(category: str):
    """Get test data for a specific category."""
    all_data = get_comprehensive_test_dataset()
    return [item for item in all_data if item.get('category') == category]

def get_balanced_evaluation_dataset(queries_per_category: int = 3):
    """Get a balanced dataset with equal representation from each category."""
    categories = ['factual', 'philosophical', 'practical', 'spiritual', 'ethical']
    balanced_data = []
    
    for category in categories:
        category_data = get_category_specific_dataset(category)
        balanced_data.extend(category_data[:queries_per_category])
    
    return balanced_data

def get_evaluation_statistics():
    """Get statistics about the test dataset."""
    all_data = get_comprehensive_test_dataset()
    categories = {}
    
    for item in all_data:
        cat = item.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        'total_queries': len(all_data),
        'categories': categories,
        'avg_relevant_docs': sum(len(item['ground_truth'].get('relevant_docs', [])) for item in all_data) / len(all_data)
    }