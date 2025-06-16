#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
朴素贝叶斯文本分类器
支持高频词特征和TF-IDF特征两种模式
"""

import os
import re
import jieba
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class NaiveBayesClassifier:
    def __init__(self, feature_method='freq', max_features=3000, use_smote=False):
        """
        初始化朴素贝叶斯分类器
        
        Args:
            feature_method (str): 特征提取方法，'freq'表示高频词，'tfidf'表示TF-IDF
            max_features (int): 最大特征数量
            use_smote (bool): 是否使用SMOTE进行样本平衡
        """
        self.feature_method = feature_method
        self.max_features = max_features
        self.use_smote = use_smote
        self.vocabulary = None
        self.class_priors = {}
        self.feature_probs = {}
        self.tfidf_vectorizer = None
        
    def load_stopwords(self, stopwords_file='stopwords.txt'):
        """加载停用词"""
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set(f.read().strip().split('\n'))
        except FileNotFoundError:
            # 默认停用词列表
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '要', '说', '来', '可以', '能', '会', '这', '那', '你', '他', '她'}
        return stopwords
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 去除非中文字符
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        # 分词
        words = jieba.cut(text)
        # 去除停用词
        stopwords = self.load_stopwords()
        words = [word for word in words if word not in stopwords and len(word) > 1]
        return words
    
    def extract_freq_features(self, texts):
        """提取高频词特征"""
        all_words = []
        processed_texts = []
        
        for text in texts:
            words = self.preprocess_text(text)
            processed_texts.append(words)
            all_words.extend(words)
        
        # 统计词频，选择高频词
        word_counts = Counter(all_words)
        vocab = [word for word, count in word_counts.most_common(self.max_features)]
        self.vocabulary = {word: idx for idx, word in enumerate(vocab)}
        
        # 构建特征矩阵
        features = np.zeros((len(texts), len(vocab)))
        for i, words in enumerate(processed_texts):
            word_count = Counter(words)
            for word, count in word_count.items():
                if word in self.vocabulary:
                    features[i, self.vocabulary[word]] = count
        
        return features
    
    def extract_tfidf_features(self, texts):
        """提取TF-IDF特征"""
        # 预处理文本
        processed_texts = []
        for text in texts:
            words = self.preprocess_text(text)
            processed_texts.append(' '.join(words))
        
        # 使用TfidfVectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 1),
                min_df=1
            )
            features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        else:
            features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        
        return features
    
    def extract_features(self, texts):
        """根据选择的方法提取特征"""
        if self.feature_method == 'freq':
            return self.extract_freq_features(texts)
        elif self.feature_method == 'tfidf':
            return self.extract_tfidf_features(texts)
        else:
            raise ValueError("feature_method must be 'freq' or 'tfidf'")
    
    def train(self, X_text, y):
        """训练朴素贝叶斯分类器"""
        # 提取特征
        X = self.extract_features(X_text)
        
        # 如果使用SMOTE进行样本平衡
        if self.use_smote and len(set(y)) > 1:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"SMOTE后样本分布: {Counter(y)}")
        
        # 计算类先验概率
        unique_classes = np.unique(y)
        total_samples = len(y)
        
        for class_label in unique_classes:
            class_count = np.sum(y == class_label)
            self.class_priors[class_label] = class_count / total_samples
        
        # 计算特征概率
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_features = X[class_mask]
            
            # 使用拉普拉斯平滑
            feature_sums = np.sum(class_features, axis=0) + 1
            total_features = np.sum(feature_sums)
            
            self.feature_probs[class_label] = feature_sums / total_features
    
    def predict(self, X_text):
        """预测"""
        X = self.extract_features(X_text)
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for class_label in self.class_priors:
                # 计算对数概率，避免下溢
                log_prob = np.log(self.class_priors[class_label])
                
                # 计算特征概率
                feature_probs = self.feature_probs[class_label]
                
                # 对于非零特征，计算概率
                for i, feature_value in enumerate(sample):
                    if feature_value > 0:
                        log_prob += feature_value * np.log(feature_probs[i])
                
                class_scores[class_label] = log_prob
            
            # 选择概率最大的类别
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)


def load_email_data(data_dir='emails'):
    """加载邮件数据"""
    texts = []
    labels = []
    
    # 加载垃圾邮件
    spam_dir = os.path.join(data_dir, 'spam')
    if os.path.exists(spam_dir):
        for filename in os.listdir(spam_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(spam_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append('spam')
                except:
                    continue
    
    # 加载正常邮件
    ham_dir = os.path.join(data_dir, 'ham')
    if os.path.exists(ham_dir):
        for filename in os.listdir(ham_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(ham_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append('ham')
                except:
                    continue
    
    # 如果没有找到邮件文件，创建示例数据
    if not texts:
        print("未找到邮件数据文件，使用示例数据...")
        texts = [
            # 垃圾邮件示例
            "恭喜您中奖了！请点击链接领取奖金！马上行动吧！",
            "免费获得iPhone最新款！立即注册！不要错过这个机会！",
            "您的账户余额不足，请及时充值，否则将停机",
            "紧急通知：您的密码即将过期，立即修改",
            "限时优惠！买一送一！全场五折大酬宾！",
            "代写毕业论文，硕博团队操作，保过查重！联系我们",
            "【XX银行】账户被冻结！立即点击链接解冻",
            "投资理财，月收益百分之三十，零风险高回报",
            "独家内幕消息，股票必涨，赶紧买入",
            "网络兼职，日赚三百，在家即可完成",
            "美女主播在线聊天，点击进入房间",
            "减肥神药，一周瘦十斤，无副作用",
            # 正常邮件示例
            "明天的会议改到下午3点，请各位同事准时参加",
            "项目进度报告已发送到您的邮箱，请查收",
            "请查收本周工作总结，有问题及时反馈",
            "生日快乐！祝您身体健康，工作顺利！",
            "感谢您的支持与配合，期待继续合作",
            "会议记录已整理完毕，请查看附件",
            "培训资料已上传至共享文件夹",
            "月度绩效考核结果已公布，请查看",
            "公司年会通知，时间地点详见附件",
            "新员工入职手续办理指南",
            "系统维护通知，请提前保存工作内容",
            "客户满意度调查报告已完成"
        ]
        labels = (['spam'] * 12) + (['ham'] * 12)
    
    return texts, labels


def evaluate_model(classifier, X_test, y_test):
    """评估模型性能"""
    y_pred = classifier.predict(X_test)
    
    print("="*50)
    print("模型评估结果")
    print("="*50)
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    return accuracy, y_pred


def main():
    """主函数"""
    print("朴素贝叶斯文本分类器")
    print("="*50)
    
    # 加载数据
    texts, labels = load_email_data()
    print(f"加载数据: {len(texts)} 条邮件")
    print(f"类别分布: {Counter(labels)}")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    # 测试不同的特征提取方法
    methods = ['freq', 'tfidf']
    results = {}
    
    for method in methods:
        print(f"\n{'='*20} {method.upper()} 特征 {'='*20}")
        
        # 不使用SMOTE
        print("不使用SMOTE:")
        classifier = NaiveBayesClassifier(feature_method=method, use_smote=False)
        classifier.train(X_train, np.array(y_train))
        accuracy, _ = evaluate_model(classifier, X_test, np.array(y_test))
        results[f'{method}_no_smote'] = accuracy
        
        # 使用SMOTE
        print("\n使用SMOTE:")
        classifier_smote = NaiveBayesClassifier(feature_method=method, use_smote=True)
        classifier_smote.train(X_train, np.array(y_train))
        accuracy_smote, _ = evaluate_model(classifier_smote, X_test, np.array(y_test))
        results[f'{method}_smote'] = accuracy_smote
    
    # 总结结果
    print("\n" + "="*50)
    print("性能对比总结")
    print("="*50)
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.4f}")


if __name__ == "__main__":
    main() 