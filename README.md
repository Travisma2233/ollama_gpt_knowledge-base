# 本地知识库系统

这是一个基于 Ollama 和 OpenRouter 的本地知识库系统，支持多种文档格式，并具有文档变更检测功能。

## 功能特点

- 支持多种文档格式（txt, md, py, docx, pdf）
- 使用 nomic-embed-text 进行文本嵌入
- 使用 OpenRouter API 进行问答
- 本地持久化存储
- 自动检测文档更新和删除
- 增量更新知识库

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 创建知识库实例

```python
kb = KnowledgeBase(storage_dir="./my_knowledge_base")
```

2. 添加/更新文档

```python
kb.add_documents_from_directory("RAG\information")
```

3. 查询知识库

```python
answer = kb.query(question)
print(answer)
```                     