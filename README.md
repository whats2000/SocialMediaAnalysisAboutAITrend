# 社群媒體 AI 趨勢分析專案 (Social Media AI Trend Analysis)

這是一個完整的自然語言處理 (NLP) 和資料科學專案，分析來自台灣 PTT 平台的 AI 相關討論。本專案包含多個深度學習和機器學習技術的實作，適合學習現代文本分析技術。

## 📊 專案資料集概覽

- **資料來源**: PTT (台灣最大的社群平台)
- **資料期間**: 2022年1月1日 - 2024年5月31日
- **資料筆數**: 2,172 筆 AI 相關討論
- **資料大小**: 186.8+ KB
- **分析主題**: AI 技術趨勢、產業應用、職涯影響等

## 🚀 分析模組與學習技術

### 1. **TF-IDF 與 N-gram 分析** (`TFIDFNGramAnalysis.ipynb`)
**學習技術**:
- **TF-IDF (詞頻-逆文件頻率)**: 學習如何量化文字重要性
- **N-gram 分析**: Bi-gram, Tri-gram 詞組分析技術
- **Cosine 相似度**: 計算文件間的相似性
- **網路視覺化**: 使用 PyVis 和 NetworkX 創建互動式網絡圖
- **中文分詞**: Jieba 分詞技術與詞典自定義

**核心技能**:
```python
- sklearn.feature_extraction.text (TfidfVectorizer, CountVectorizer)
- networkx (網絡分析)
- pyvis (互動式視覺化)
- jieba (中文自然語言處理)
```

### 2. **BERT 情感分析** (`BertSentimentAnalysis.ipynb`)
**學習技術**:
- **BERT 模型**: 使用預訓練的 Transformer 模型
- **情感分類**: 正面、負面、中性情感判斷
- **模型微調**: 在中文資料上的遷移學習
- **批次處理**: 大量文本的高效處理
- **主題建模**: BERTopic 無監督主題發現

**核心技能**:
```python
- transformers (HuggingFace)
- torch (PyTorch 深度學習)
- sentence_transformers (句子嵌入)
- bertopic (主題建模)
- datasets (資料集處理)
```

### 3. **命名實體識別 (NER)** (`BertNERAnalysis.ipynb`)
**學習技術**:
- **CKIP Transformers**: 台灣中研院的中文 NLP 工具
- **實體抽取**: 人名、地名、組織名等實體識別
- **詞性標註**: 分析詞彙的語法功能
- **實體關係分析**: 分析不同實體間的關聯性

**核心技能**:
```python
- ckip_transformers (中文 NLP)
- 實體識別技術
- 詞性分析
- 中文語言模型應用
```

### 4. **LDA 主題建模** (`LDATopicModel.ipynb`)
**學習技術**:
- **潛在狄利克雷分配 (LDA)**: 無監督主題發現
- **主題一致性評估**: Coherence Score 計算
- **超參數調優**: 最佳主題數量選擇
- **互動式主題視覺化**: pyLDAvis 動態圖表
- **多核心處理**: 大資料的平行處理

**核心技能**:
```python
- gensim (主題建模)
- pyLDAvis (主題視覺化)
- multiprocessing (平行處理)
- 統計學習方法
```

### 5. **社群網路分析** (`NetworkAnalysis.ipynb`)
**學習技術**:
- **共現網路**: 詞彙共同出現關係分析
- **網路指標**: 中心性、聚類係數、路徑長度
- **社群偵測**: 找出詞彙群組和話題社群
- **動態網路視覺化**: 互動式網路圖製作
- **影響力分析**: 識別關鍵節點和橋接詞彙

**核心技能**:
```python
- networkx (網路分析)
- pyvis (網路視覺化)
- 圖論算法
- 社群偵測算法
```

### 6. **詞典式情感分析** (`LexiconSentiment.ipynb`)
**學習技術**:
- **LIWC 詞典**: 心理語言學分析工具
- **自定義詞典**: 建立領域特定的情感詞典
- **情感極性計算**: 基於詞典的情感強度量化
- **詞雲視覺化**: 情感詞彙的視覺化呈現

**核心技能**:
```python
- 詞典式分析方法
- 心理語言學分析
- wordcloud (詞雲製作)
- 情感計算技術
```

### 7. **SnowNLP 情感分析** (`SnowNLPSentiment.ipynb`)
**學習技術**:
- **SnowNLP**: 中文自然語言處理套件
- **情感概率**: 計算文本正面情感的機率
- **時間序列分析**: 情感趨勢的時間變化
- **CKIP 工具整合**: 多種中文 NLP 工具組合使用

**核心技能**:
```python
- snownlp (中文情感分析)
- ckiptagger (中文標記)
- 時間序列分析
- 多工具整合技術
```

### 8. **關鍵詞分析** (`KeyWordAnalysis.ipynb`)
**學習技術**:
- **TF-IDF 關鍵詞提取**: 自動識別重要詞彙
- **TextRank 算法**: 基於圖的關鍵詞排序
- **詞頻統計**: 統計分析和趨勢識別
- **關鍵詞共現**: 分析詞彙間的關聯模式

**核心技能**:
```python
- jieba.analyse (中文關鍵詞提取)
- TextRank 算法
- 統計分析方法
- 關聯規則挖掘
```

## 🛠️ 技術棧與工具

### 深度學習框架
- **PyTorch**: 深度學習模型訓練與推理
- **HuggingFace Transformers**: 預訓練模型使用
- **Sentence Transformers**: 句子嵌入技術

### 機器學習庫
- **Scikit-learn**: 傳統機器學習算法
- **Gensim**: 主題建模與詞向量
- **HDBSCAN**: 層次聚類算法

### 中文 NLP 工具
- **Jieba**: 中文分詞與關鍵詞提取
- **CKIP Transformers**: 中研院中文 NLP 工具
- **SnowNLP**: 中文情感分析

### 資料視覺化
- **Matplotlib & Seaborn**: 統計圖表製作
- **PyVis**: 互動式網路視覺化
- **WordCloud**: 詞雲製作
- **pyLDAvis**: 主題模型視覺化

### 資料處理
- **Pandas**: 資料操作與分析
- **NumPy**: 數值計算
- **NetworkX**: 網路分析

## 📚 學習路徑建議

### 初學者路徑
1. **KeyWordAnalysis.ipynb** - 學習基礎文本處理
2. **LexiconSentiment.ipynb** - 理解詞典式分析
3. **SnowNLPSentiment.ipynb** - 中文情感分析入門

### 進階路徑
4. **TFIDFNGramAnalysis.ipynb** - 掌握 TF-IDF 和 N-gram
5. **LDATopicModel.ipynb** - 學習主題建模
6. **NetworkAnalysis.ipynb** - 網路分析技術

### 專家路徑
7. **BertNERAnalysis.ipynb** - 深度學習實體識別
8. **BertSentimentAnalysis.ipynb** - BERT 模型應用

## 🎯 實務應用場景

- **市場研究**: 分析消費者對產品/服務的意見
- **品牌監控**: 追蹤品牌在社群媒體的聲譽
- **趨勢分析**: 識別新興話題和技術趨勢
- **客戶洞察**: 理解客戶需求和痛點
- **競爭分析**: 分析競爭對手的市場表現

## 🔧 環境建置

確保安裝以下主要套件：
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn gensim
pip install torch transformers sentence-transformers
pip install jieba snownlp
pip install networkx pyvis
pip install ckip-transformers
pip install bertopic pyldavis
pip install wordcloud tqdm
```

## 📈 專案成果

本專案提供完整的社群媒體文本分析流程，從資料預處理到進階分析，涵蓋現代 NLP 的主要技術。透過實際的台灣 AI 討論資料，學習者可以掌握：

- 🔍 **文本探索**: 從原始文本中發現有價值的資訊
- 🤖 **AI 應用**: 實際運用最新的 NLP 模型
- 📊 **資料視覺化**: 將分析結果以直觀的方式呈現
- 🧠 **洞察發現**: 從數據中提取商業和學術價值

這個專案不僅是學習 NLP 技術的絕佳資源，也是了解台灣 AI 發展趨勢的重要參考。
