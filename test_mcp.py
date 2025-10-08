#!/usr/bin/env python3
"""Test script for MCP integration"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_mcp_call():
    """Test MCP server call"""
    try:
        # Test FSSAI lookup
        mcp_url = "https://mcp-server-agent1.onrender.com/api/producer/fssai"
        fssai_number = "20819019000744"  # From the sample data

        print(f"Testing FSSAI lookup: {mcp_url}")
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
                pin = data.get('pin')
                print(f"PIN: {pin}")
                print()

                # Test PIN lookup with the retrieved PIN
                if pin:
                    test_pin_lookup(pin)
            else:
                print(f"Message: {result.get('message')}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

def test_pin_lookup(pin):
    """Test PIN-based lookup"""
    try:
        mcp_url = "https://mcp-server-agent1.onrender.com/api/producer/pin"

        print(f"Testing PIN lookup: {mcp_url}")
        print(f"PIN: {pin}")

        response = requests.post(mcp_url, json={"pin": pin}, timeout=10)
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
                print(f"PIN: {data.get('pin')}")
            else:
                print(f"Message: {result.get('message')}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_mcp_call()