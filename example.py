# 导入 KnowledgeBase 类
from knowledge_base import KnowledgeBase

# 创建知识库实例
kb = KnowledgeBase(storage_dir="./my_knowledge_base")

# 添加/更新文档
kb.add_documents_from_directory(r"C:\Users\xinru\OneDrive\桌面\RAG\information")

# 查询知识库
question = "火星的温度？"
answer = kb.query(question)
print(answer) 