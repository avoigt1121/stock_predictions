#!/usr/bin/env python3
"""
Test script for the Stock Prediction API
Run this to test your API endpoints
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """Test single stock prediction"""
    print("\nTesting single prediction...")
    try:
        data = {
            "ticker": "AAPL",
            "period": "1y"
        }
        response = requests.post(f"{API_BASE}/predict", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch predictions"""
    print("\nTesting batch prediction...")
    try:
        data = {
            "tickers": ["AAPL", "MSFT", "TSLA"],
            "period": "1y"
        }
        response = requests.post(f"{API_BASE}/predict/batch", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_ticker():
    """Test with invalid ticker"""
    print("\nTesting invalid ticker...")
    try:
        data = {
            "ticker": "INVALID123",
            "period": "1y"
        }
        response = requests.post(f"{API_BASE}/predict", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True  # We expect this to fail gracefully
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Stock Prediction API Test Suite")
    print("=" * 40)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE}/")
            if response.status_code == 200:
                print("API is ready!")
                break
        except:
            pass
        
        if i == max_retries - 1:
            print("API is not responding. Make sure it's running on localhost:8000")
            return
        
        time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Ticker", test_invalid_ticker),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*40}")
    print("TEST SUMMARY:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nPassed: {total_passed}/{len(results)}")

if __name__ == "__main__":
    main()
