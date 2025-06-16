# 朴素贝叶斯文本分类器

## 📖 项目概述

本项目实现了一个功能完善的朴素贝叶斯文本分类器，主要用于垃圾邮件检测。项目基于经典的朴素贝叶斯算法，支持多种特征提取方法，具备样本平衡处理和多维度评估能力。

### 🎯 主要特性

- ✅ **双模式特征提取**：支持高频词特征和TF-IDF特征
- ✅ **样本平衡处理**：集成SMOTE过采样技术
- ✅ **多维度评估**：提供准确率、精度、召回率、F1值等评估指标
- ✅ **中文文本处理**：基于jieba分词的中文文本预处理
- ✅ **参数化配置**：灵活的特征方法切换机制

## 🗂️ 项目结构

```
├── classify.py           # 主实现文件（朴素贝叶斯分类器）
├── classify.ipynb        # Jupyter Notebook版本
├── requirements.txt      # 项目依赖包
├── README.md            # 项目文档
└── emails/              # 邮件数据目录（可选）
    ├── spam/            # 垃圾邮件样本
    └── ham/             # 正常邮件样本
```

## 🔧 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖包

```
jieba==0.42.1              # 中文分词
numpy==1.21.0              # 数值计算
pandas==1.3.0              # 数据处理
scikit-learn==1.0.2        # 机器学习算法
imbalanced-learn==0.8.1    # 样本平衡处理
matplotlib==3.4.2          # 数据可视化
seaborn==0.11.1            # 统计图表
```

## 🚀 快速开始

### 方式1：运行Python脚本

```bash
python classify.py
```

### 方式2：使用Jupyter Notebook

```bash
jupyter notebook classify.ipynb
```

## 🔄 特征模式切换

### 高频词特征模式

```python
# 使用高频词特征
classifier = NaiveBayesClassifier(feature_method='freq')
```

### TF-IDF特征模式

```python
# 使用TF-IDF特征
classifier = NaiveBayesClassifier(feature_method='tfidf')
```

### 完整配置示例

```python
# 高频词特征 + SMOTE样本平衡
classifier_freq = NaiveBayesClassifier(
    feature_method='freq',
    max_features=3000,
    use_smote=True
)

# TF-IDF特征 + 无样本平衡
classifier_tfidf = NaiveBayesClassifier(
    feature_method='tfidf',
    max_features=5000,
    use_smote=False
)
```

## 💡 使用示例

### 基本使用流程

```python
from classify import NaiveBayesClassifier, load_email_data, evaluate_model
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 加载数据
texts, labels = load_email_data()

# 2. 数据集分割
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 3. 创建并训练分类器
classifier = NaiveBayesClassifier(feature_method='tfidf', use_smote=True)
classifier.train(X_train, np.array(y_train))

# 4. 模型评估
accuracy, predictions = evaluate_model(classifier, X_test, np.array(y_test))

# 5. 预测新文本
new_texts = ["恭喜您中奖了！", "明天开会通知"]
predictions = classifier.predict(new_texts)
print(predictions)  # ['spam', 'ham']
```

### 批量测试不同特征方法

```python
methods = ['freq', 'tfidf']
results = {}

for method in methods:
    for use_smote in [False, True]:
        classifier = NaiveBayesClassifier(
            feature_method=method, 
            use_smote=use_smote
        )
        classifier.train(X_train, y_train)
        accuracy, _ = evaluate_model(classifier, X_test, y_test)
        
        key = f"{method}_{'smote' if use_smote else 'no_smote'}"
        results[key] = accuracy

# 性能对比
for method, acc in results.items():
    print(f"{method}: {acc:.4f}")
```

## 📊 运行结果

运行 `python classify.py` 的输出示例：

```
朴素贝叶斯文本分类器
==================================================
加载数据: 24 条邮件
类别分布: Counter({'spam': 12, 'ham': 12})
训练集: 19 条
测试集: 5 条

==================== FREQ 特征 ====================
不使用SMOTE:
准确率: 0.6000

使用SMOTE:
SMOTE后样本分布: Counter({'ham': 10, 'spam': 10})
准确率: 0.6000

==================== TFIDF 特征 ====================
不使用SMOTE:
准确率: 1.0000

分类报告:
              precision    recall  f1-score   support
         ham       1.00      1.00      1.00         3
        spam       1.00      1.00      1.00         2
    accuracy                           1.00         5

==================================================
性能对比总结
==================================================
freq_no_smote: 0.6000
freq_smote: 0.6000
tfidf_no_smote: 1.0000  ⭐ 最佳性能
tfidf_smote: 0.6000
```

## 🧠 算法原理

### 朴素贝叶斯分类器

基于贝叶斯定理，核心公式为：

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

其中：
- $P(C|X)$：给定特征向量 $X$ 下类别 $C$ 的后验概率
- $P(X|C)$：给定类别 $C$ 下特征向量 $X$ 的似然概率  
- $P(C)$：类别 $C$ 的先验概率
- $P(X)$：特征向量 $X$ 的边际概率

### 特征提取方法

#### 高频词特征

统计词频，选取高频词作为特征：

$$\text{Feature}_{word} = \text{count}(word)$$

#### TF-IDF特征

考虑词在文档中的重要性：

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

其中：
- $\text{TF}(t,d) = \frac{\text{count}(t,d)}{\sum_{t' \in d} \text{count}(t',d)}$
- $\text{IDF}(t) = \log\frac{|D|}{|\{d \in D : t \in d\}|}$

### 拉普拉斯平滑

避免零概率问题：

$$P(w_i|c) = \frac{\text{count}(w_i, c) + 1}{\sum_w \text{count}(w, c) + |V|}$$

## 📈 评估指标

### 准确率 (Accuracy)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 精度 (Precision)  
$$\text{Precision} = \frac{TP}{TP + FP}$$

### 召回率 (Recall)
$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1值
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

## 🔍 性能优化

### SMOTE样本平衡

解决类别不平衡问题：

```python
# 启用SMOTE过采样
classifier = NaiveBayesClassifier(use_smote=True)
```

**效果**：
- 生成合成少数类样本
- 提升模型在少数类上的召回率
- 缓解样本不平衡导致的偏向问题

### 特征选择优化

```python
# 调整特征数量
classifier = NaiveBayesClassifier(
    feature_method='tfidf',
    max_features=5000  # 增加特征维度
)
```

## 🎯 应用场景

1. **垃圾邮件检测**：识别垃圾邮件和正常邮件
2. **情感分析**：分析文本情感倾向（积极/消极/中性）
3. **主题分类**：为新闻、文章进行主题标签
4. **意图识别**：在对话系统中理解用户意图  
5. **诈骗检测**：识别钓鱼邮件、虚假信息

## 📝 自定义数据

### 数据格式

在 `emails/` 目录下创建以下结构：

```
emails/
├── spam/           # 垃圾邮件文件夹
│   ├── email1.txt
│   └── email2.txt
└── ham/            # 正常邮件文件夹
    ├── email1.txt
    └── email2.txt
```

### 示例文件内容

**垃圾邮件示例** (`emails/spam/example1.txt`)：
```
恭喜您中奖了！立即点击链接领取100万元大奖！
```

**正常邮件示例** (`emails/ham/example1.txt`)：
```
明天下午3点在会议室A举行项目会议，请准时参加。
```

## 🔧 扩展功能

### 自定义停用词

```python
# 创建停用词文件
with open('stopwords.txt', 'w', encoding='utf-8') as f:
    f.write('的\n了\n在\n是\n我\n')  # 每行一个停用词
```

### 批量预测

```python
batch_texts = ["文本1", "文本2", "文本3"]
predictions = classifier.predict(batch_texts)
for text, pred in zip(batch_texts, predictions):
    print(f"'{text}' -> {pred}")
```

## 📋 结果分析

根据测试结果：

- **TF-IDF特征** 表现优于高频词特征
- **无SMOTE** 在小数据集上效果更好  
- TF-IDF能更好地捕捉文本语义信息
- 建议生产环境使用 `feature_method='tfidf'`

## 📄 许可证

MIT License

## 👨‍💻 作者

[您的姓名]

---

*本项目完成了NLP文本分类的核心功能，包括特征提取、样本平衡、模型评估等关键技术，适合作为文本分类学习和应用的参考实现。* 