#!/usr/bin/env python3
"""
Advanced script to test the RAG query endpoint with multiple options
"""

import requests
import json
import sys
from typing import Dict, Any

class RAGQueryTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def query(self, question: str) -> Dict[Any, Any]:
        """Send a query to the RAG system"""
        url = f"{self.base_url}/query"
        
        payload = {"question": question}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            response = self.session.post(url, data=payload, headers=headers, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making request: {e}")
            return {"error": str(e)}
    
    def test_multiple_questions(self):
        """Test multiple sample questions"""
        questions = [
            "What is Retrieval-Augmented Generation?",
            "How does RAG improve answer accuracy?",
            "Explain the difference between traditional QA and RAG systems"
        ]
        
        print("Testing multiple questions...\n")
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            result = self.query(question)
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Response: {result.get('answer', 'No answer provided')}")
            print()

def main():
    """Main function to run the script"""
    if len(sys.argv) > 1:
        # If argument provided, use it as question
        question = sys.argv[1]
        tester = RAGQueryTester()
        result = tester.query(question)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        else:
            print(json.dumps(result, indent=2))
    else:
        # Run with sample questions
        tester = RAGQueryTester()
        tester.test_multiple_questions()

if __name__ == "__main__":
    main()#!/usr/bin/env python3

    '''
    # Test with specific question
    python advanced_test.py "What is Retrieval-Augmented Generation?"
    # Test with default sample questions
    python advanced_test.py
    '''