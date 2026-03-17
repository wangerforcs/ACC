#!/usr/bin/env python3
"""
Test data for object extractor
"""

# Comprehensive test cases
test_data = [
    # Basic cases
    {
        "instruction": "Pick up the black bowl between the plate and the ramekin and place it on the plate.",
        "expected": ["black bowl between the plate and the ramekin", "plate"]
    },
    {
        "instruction": "Put the book on the table.",
        "expected": ["book", "table"]
    },
    {
        "instruction": "Move the cup to the shelf.",
        "expected": ["cup", "shelf"]
    },
    
    # Multiple modifiers
    {
        "instruction": "Take the small red plastic cup from the kitchen counter and put it in the blue drawer.",
        "expected": ["small red plastic cup", "blue drawer"]
    },
    {
        "instruction": "Place the round wooden table next to the window in the living room.",
        "expected": ["round wooden table next to the window", "living room"]
    },
    
    # Various prepositions
    {
        "instruction": "Put the cat under the bed.",
        "expected": ["cat", "bed"]
    },
    {
        "instruction": "Move the plant over the sofa.",
        "expected": ["plant", "sofa"]
    },
    {
        "instruction": "Place the vase behind the TV.",
        "expected": ["vase", "tv"]
    },
    {
        "instruction": "Set the keys beside the phone.",
        "expected": ["keys", "phone"]
    },
    
    # Complex contexts
    {
        "instruction": "Pick up the silver fork between the knife and the spoon and place it on the napkin.",
        "expected": ["silver fork between the knife and the spoon", "napkin"]
    },
    {
        "instruction": "Move the large blue box under the stairs into the garage.",
        "expected": ["large blue box under the stairs", "garage"]
    },
    
    # Multiple actions
    {
        "instruction": "Take the bread from the fridge, put it in the toaster, then place the toast on the plate.",
        "expected": ["bread", "toaster", "toast", "plate"]
    },
    {
        "instruction": "Grab the pen from the desk, write a note, and put the note in the envelope.",
        "expected": ["pen", "note", "envelope"]
    },
    
    # Ambiguous cases
    {
        "instruction": "Move the box to the room with the red door.",
        "expected": ["box", "room"]
    },
    {
        "instruction": "Place the painting above the fireplace in the dining room.",
        "expected": ["painting", "fireplace"]
    }
]

# Function to get test data
def get_test_data():
    return test_data

# Function to add custom test cases
def add_test_case(instruction, expected):
    test_data.append({
        "instruction": instruction,
        "expected": expected
    })

if __name__ == "__main__":
    print("Test Data Overview:")
    print(f"Total test cases: {len(test_data)}")
    print()
    
    for i, test_case in enumerate(test_data[:5]):  # Show first 5 cases
        print(f"Case {i+1}:")
        print(f"Instruction: {test_case['instruction']}")
        print(f"Expected:    {test_case['expected']}")
        print()
    
    if len(test_data) > 5:
        print(f"... and {len(test_data) - 5} more cases")