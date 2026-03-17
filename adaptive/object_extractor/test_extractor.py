#!/usr/bin/env python3
"""
Test script for object extractor
"""

import json
from extractor import extract_objects, extract_objects_with_details

# Load test data from JSON file
def load_test_data():
    """
    Load test data from test_data.json
    """
    try:
        with open('test_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: test_data.json not found")
        return []

# Test cases (fallback if JSON fails)
test_cases = [
    {
        "instruction": "Pick up the black bowl between the plate and the ramekin and place it on the plate.",
        "expected": ["black bowl between the plate and the ramekin", "plate"]
    },
    {
        "instruction": "Put the red cup next to the blue book on the table.",
        "expected": ["red cup next to the blue book", "table"]
    },
    {
        "instruction": "Move the small green box under the bed into the closet.",
        "expected": ["small green box under the bed", "closet"]
    },
    {
        "instruction": "Take the silver key from the drawer and insert it into the lock.",
        "expected": ["silver key", "lock"]
    },
    {
        "instruction": "Place the white plate over the bowl on the counter.",
        "expected": ["white plate", "bowl"]
    }
]

def test_extractor():
    """
    Test the object extractor function
    """
    # Load test data from JSON
    json_test_data = load_test_data()
    
    # Use JSON data if available, otherwise use fallback
    test_data = json_test_data if json_test_data else test_cases
    
    print("Testing Object Extractor...")
    print("=" * 60)
    
    total = len(test_data)
    correct = 0
    
    for i, test_case in enumerate(test_data):
        instruction = test_case["instruction"]
        expected = test_case.get("expected_objects", test_case.get("expected", []))
        
        result = extract_objects(instruction)
        
        # Check if all expected objects are present (order doesn't matter)
        is_correct = set(result) == set(expected)
        
        if is_correct:
            correct += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"Test {i+1}: {status}")
        print(f"Instruction: {instruction}")
        print(f"Expected:    {expected}")
        print(f"Got:         {result}")
        print()
    
    print("=" * 60)
    print(f"Results: {correct}/{total} tests passed")
    print(f"Accuracy: {correct/total:.2%}")
    print("=" * 60)


def test_detailed_extractor():
    """
    Test the detailed object extractor function
    """
    # Load test data from JSON
    json_test_data = load_test_data()
    
    # Use JSON data if available, otherwise use fallback
    test_data = json_test_data if json_test_data else test_cases
    
    print("\nTesting Detailed Object Extractor...")
    print("=" * 60)
    
    for test_case in test_data[:5]:  # Show first 5 cases
        instruction = test_case["instruction"]
        result = extract_objects_with_details(instruction)
        
        print(f"Instruction: {instruction}")
        print(f"Extracted:   {result['extracted_objects']}")
        print()

if __name__ == "__main__":
    test_extractor()
    test_detailed_extractor()