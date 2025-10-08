#!/usr/bin/env python3
"""Test script for MCP integration"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_mcp_call():
    """Test MCP server call"""
    try:
        mcp_url = "https://mcp-server-agent1.onrender.com/api/producer/fssai"
        fssai_number = "20819019000744"  # From the sample data

        print(f"Testing MCP call to: {mcp_url}")
        print(f"FSSAI Number: {fssai_number}")

        response = requests.post(mcp_url, json={"fssai_number": fssai_number}, timeout=10)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("Success!")
            print(f"Status: {result.get('status')}")
            if result.get('status') == 'success':
                data = result.get('data', {})
                print(f"Producer Name: {data.get('name')}")
                address = data.get('address', 'N/A')
                try:
                    print(f"Address: {address}")
                except UnicodeEncodeError:
                    print(f"Address: {address.encode('utf-8', errors='replace').decode('utf-8')}")
                print(f"FSSAI: {data.get('fssai_license_number')}")
            else:
                print(f"Message: {result.get('message')}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_mcp_call()