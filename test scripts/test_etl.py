#!/usr/bin/env python3
"""
Advanced script to test the RAG PDF ingestion endpoint with multiple options
"""

import requests
import json
import sys
import os
from typing import Dict, Any
from pathlib import Path

class RAGIngestionTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def ingest_pdf(self, file_path: str) -> Dict[Any, Any]:
        """Send a PDF file for ingestion to the RAG system"""
        url = f"{self.base_url}/ingest/pdf"
        
        # Validate file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Prepare file for upload
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            headers = {
                "accept": "application/json"
            }
            
            try:
                response = self.session.post(url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error making request: {e}")
                return {"error": str(e)}
    
    def ingest_docx(self, file_path: str) -> Dict[Any, Any]:
        """Send a DOCX file for ingestion to the RAG system"""
        url = f"{self.base_url}/ingest/docx"
        
        # Validate file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Prepare file for upload
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
            headers = {
                "accept": "application/json"
            }
            
            try:
                response = self.session.post(url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error making request: {e}")
                return {"error": str(e)}
    
    def ingest_web(self, url: str) -> Dict[Any, Any]:
        """Send a web URL for ingestion to the RAG system"""
        url_endpoint = f"{self.base_url}/ingest/web"
        
        payload = {"url": url}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            response = self.session.post(url_endpoint, data=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making request: {e}")
            return {"error": str(e)}
    
    def test_pdf_ingestion(self, file_paths: list):
        """Test multiple PDF files for ingestion"""
        print("Testing PDF ingestion...\n")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"PDF {i}: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                print(f"  ❌ Error: File not found")
                continue
                
            result = self.ingest_pdf(file_path)
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Success: {result.get('message', 'Ingestion started')}")
                if 'file_path' in result:
                    print(f"  File saved to: {result['file_path']}")
            print()
    
    def test_docx_ingestion(self, file_paths: list):
        """Test multiple DOCX files for ingestion"""
        print("Testing DOCX ingestion...\n")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"DOCX {i}: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                print(f"  ❌ Error: File not found")
                continue
                
            result = self.ingest_docx(file_path)
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Success: {result.get('message', 'Ingestion started')}")
                if 'file_path' in result:
                    print(f"  File saved to: {result['file_path']}")
            print()
    
    def test_web_ingestion(self, urls: list):
        """Test multiple web URLs for ingestion"""
        print("Testing web URL ingestion...\n")
        
        for i, url in enumerate(urls, 1):
            print(f"Web URL {i}: {url}")
            
            result = self.ingest_web(url)
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Success: {result.get('message', 'Ingestion started')}")
                if 'file_path' in result:
                    print(f"  Source URL: {result['file_path']}")
            print()

def main():
    """Main function to run the script"""
    tester = RAGIngestionTester()
    
    # Test cases
    pdf_files = [
        "sample.pdf",  # Replace with actual PDF file paths
        "document.pdf"
    ]
    
    docx_files = [
        "sample.docx",  # Replace with actual DOCX file paths
        "document.docx"
    ]
    
    web_urls = [
        "https://example.com/page",  # Sample URL from knowledge base
        "https://openrouter.ai/api/v1"  # Another sample URL
    ]
    
    # If arguments provided, use them as specific test files
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        
        if action == "pdf":
            if len(sys.argv) > 2:
                # Specific PDF file test
                file_path = sys.argv[2]
                result = tester.ingest_pdf(file_path)
                print(json.dumps(result, indent=2))
            else:
                print("Usage: python test_ingestion.py pdf <file_path>")
                
        elif action == "docx":
            if len(sys.argv) > 2:
                # Specific DOCX file test
                file_path = sys.argv[2]
                result = tester.ingest_docx(file_path)
                print(json.dumps(result, indent=2))
            else:
                print("Usage: python test_ingestion.py docx <file_path>")
                
        elif action == "web":
            if len(sys.argv) > 2:
                # Specific web URL test
                url = sys.argv[2]
                result = tester.ingest_web(url)
                print(json.dumps(result, indent=2))
            else:
                print("Usage: python test_ingestion.py web <url>")
                
        else:
            print("Usage: python test_ingestion.py [pdf|docx|web] [file_path|url]")
            
    else:
        # Run comprehensive tests
        print("🧪 Running comprehensive ingestion tests...\n")
        
        # Test PDF ingestion
        tester.test_pdf_ingestion(pdf_files)
        
        # Test DOCX ingestion
        tester.test_docx_ingestion(docx_files)
        
        # Test web ingestion
        tester.test_web_ingestion(web_urls)

if __name__ == "__main__":
    main()


    '''
    python test_ingestion.py
    python test_ingestion.py pdf sample.pdf
    python test_ingestion.py docx sample.docx
    python test_ingestion.py web https://example.com/page
    '''