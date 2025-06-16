#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dev_skills_env.py - NumPy矩阵运算示例
使用虚拟环境nlp_yyr和NumPy 1.22.3版本
"""

import numpy as np

def main():
    print("=" * 50)
    print("NumPy矩阵运算示例")
    print("虚拟环境: nlp_yyr")
    print(f"NumPy版本: {np.__version__}")
    print("=" * 50)
    
    # 创建两个示例矩阵
    print("\n1. 创建矩阵")
    matrix_a = np.array([[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9]])
    matrix_b = np.array([[9, 8, 7], 
                         [6, 5, 4], 
                         [3, 2, 1]])
    
    print("矩阵 A:")
    print(matrix_a)
    print("\n矩阵 B:")
    print(matrix_b)
    
    # 矩阵加法
    print("\n2. 矩阵加法 (A + B):")
    result_add = matrix_a + matrix_b
    print(result_add)
    
    # 矩阵乘法
    print("\n3. 矩阵乘法 (A @ B):")
    result_mul = matrix_a @ matrix_b
    print(result_mul)
    
    # 矩阵转置
    print("\n4. 矩阵A的转置:")
    result_transpose = matrix_a.T
    print(result_transpose)
    
    # 矩阵的行列式
    print("\n5. 矩阵A的行列式:")
    det_a = np.linalg.det(matrix_a)
    print(f"det(A) = {det_a}")
    
    # 创建特殊矩阵
    print("\n6. 特殊矩阵创建:")
    zeros_matrix = np.zeros((3, 3))
    ones_matrix = np.ones((3, 3))
    identity_matrix = np.eye(3)
    
    print("零矩阵:")
    print(zeros_matrix)
    print("\n单位矩阵:")
    print(identity_matrix)
    
    # 随机矩阵
    print("\n7. 随机矩阵 (3x3):")
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    random_matrix = np.random.rand(3, 3)
    print(random_matrix)
    
    # 矩阵统计
    print("\n8. 矩阵统计信息:")
    print(f"矩阵A的最大值: {np.max(matrix_a)}")
    print(f"矩阵A的最小值: {np.min(matrix_a)}")
    print(f"矩阵A的平均值: {np.mean(matrix_a):.2f}")
    print(f"矩阵A的标准差: {np.std(matrix_a):.2f}")
    
    return "矩阵运算完成！"

if __name__ == "__main__":
    result = main()
    print(f"\n{result}") 