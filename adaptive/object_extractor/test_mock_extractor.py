#!/usr/bin/env python3
"""
测试Mock Object Extractor
验证其是否能够为libero.txt中的每个指令生成正确的结果
"""

from mock_extractor import extract_object_roles, extract_objects, extract_objects_with_details

# 读取libero.txt文件
def read_libero_file():
    with open('libero.txt', 'r', encoding='utf-8') as f:
        instructions = [line.strip() for line in f if line.strip()]
    return instructions

# 测试mock extractor
def test_mock_extractor():
    print("测试Mock Object Extractor...")
    print("=" * 80)
    
    # 读取指令
    instructions = read_libero_file()
    total = len(instructions)
    passed = 0
    
    for i, instruction in enumerate(instructions):
        # 测试extract_objects函数
        result = extract_objects(instruction)
        
        # 测试extract_objects_with_details函数
        detailed_result = extract_objects_with_details(instruction)
        role_result = extract_object_roles(instruction)
        
        # 验证结果
        is_correct = len(result) > 0
        detailed_is_correct = detailed_result['instruction'] == instruction and detailed_result['extracted_objects'] == result
        role_is_correct = role_result["instruction"] == instruction and role_result["all_objects"] == result
        
        if is_correct and detailed_is_correct and role_is_correct:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"Test {i+1}: {status}")
        print(f"Instruction: {instruction}")
        print(f"Extracted:   {result}")
        print(f"Detailed:    {detailed_result}")
        print(f"Roles:       {role_result}")
        print()
    
    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print(f"Accuracy: {passed/total:.2%}")
    print("=" * 80)

if __name__ == "__main__":
    test_mock_extractor()
