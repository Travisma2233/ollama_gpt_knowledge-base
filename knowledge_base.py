from typing import List, Dict
import ollama
import numpy as np
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
import json
import requests
import pickle

class KnowledgeBase:
    def __init__(self, storage_dir: str = "./kb_storage"):
        """初始化知识库
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.documents_file = self.storage_dir / "documents.json"
        self.embeddings_file = self.storage_dir / "embeddings.pkl"
        self.metadata_file = self.storage_dir / "metadata.json"  # 新增：存储文件元数据
        
        # 加载元数据
        self.metadata = self.load_metadata()
        # 加载已存储的数据
        self.documents = self.load_documents()
        self.embeddings = self.load_embeddings()
        
    def load_documents(self) -> List[str]:
        """加载已存储的文档"""
        if self.documents_file.exists():
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def load_embeddings(self) -> List[List[float]]:
        """加载已存储的嵌入向量"""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                return pickle.load(f)
        return []
    
    def load_metadata(self) -> Dict:
        """加载文件元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_data(self):
        """保存数据到文件"""
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def save_metadata(self):
        """保存文件元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def add_document(self, content: str):
        """添加文档到知识库"""
        # 获取文档的嵌入向量
        response = ollama.embeddings(
            model='nomic-embed-text',
            prompt=content
        )
        embedding = response['embedding']
        
        self.documents.append(content)
        self.embeddings.append(embedding)
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索最相关的文档"""
        # 获取查询的嵌入向量
        query_embedding = ollama.embeddings(
            model='nomic-embed-text',
            prompt=query
        )['embedding']
        
        # 计算相似度
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append(similarity)
            
        # 获取最相关的文档索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'similarity': similarities[idx]
            })
            
        return results
    
    def query(self, question: str) -> str:
        """使用 OpenRouter API 查询知识库并生成回答"""
        relevant_docs = self.search(question)
        context = "\n".join([doc['content'] for doc in relevant_docs])
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        payload = {
            "model": "openai/gpt-4o-mini",  # 可以选择其他模型
            "messages": [
                {
                    "role": "user",
                    "content": f"""基于以下上下文回答问题:

上下文:
{context}

问题: {question}

请基于上述上下文提供准确的回答。如果上下文中没有相关信息，请说明无法回答。"""
                }
            ]
        }
        
        headers = {
            "Authorization": "Bearer sk-",  # 替换为您的 OpenRouter API key
            "HTTP-Referer": "http://localhost:3000",  # 您的域名
            "X-Title": "Knowledge Base App",  # 您的应用名称
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        
        return response_data['choices'][0]['message']['content']

    def read_file_content(self, file_path: Path) -> str:
        """读取不同格式文件的内容
        
        Args:
            file_path: 文件路径
        Returns:
            str: 文件内容
        """
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md', '.py']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif suffix == '.docx':
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
        elif suffix == '.pdf':
            reader = PdfReader(file_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text
            
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def check_file_changes(self, file_path: Path) -> bool:
        """检查文件是否发生变化
        
        Returns:
            bool: True 如果文件是新的或已修改
        """
        relative_path = str(file_path)
        mtime = file_path.stat().st_mtime
        
        if relative_path not in self.metadata:
            return True
        
        return mtime != self.metadata[relative_path]['mtime']

    def add_documents_from_directory(self, directory_path: str, file_extensions: List[str] = ['.txt', '.md', '.py', '.docx', '.pdf']):
        """从指定目录添加所有文档到知识库"""
        directory = Path(directory_path)
        
        # 检查已存在文件的变化
        current_files = set()
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in file_extensions:
                abs_path = str(file_path.absolute())
                current_files.add(abs_path)
                
                if self.check_file_changes(file_path):
                    try:
                        content = self.read_file_content(file_path)
                        relative_path = str(file_path.relative_to(directory))
                        document = f"文件: {relative_path}\n\n{content}"
                        
                        # 如果文件已存在，更新它
                        doc_index = None
                        for i, doc in enumerate(self.documents):
                            doc_path = doc.split('\n\n')[0].replace('文件: ', '')
                            if doc_path == relative_path:  # 使用相对路径比较
                                doc_index = i
                                break
                        
                        if doc_index is not None:
                            self.documents[doc_index] = document
                            # 更新嵌入向量
                            response = ollama.embeddings(
                                model='nomic-embed-text',
                                prompt=content
                            )
                            self.embeddings[doc_index] = response['embedding']
                            print(f"已更新文件: {relative_path}")
                        else:
                            # 添加新文件
                            self.add_document(document)
                            print(f"已添加文件: {relative_path}")
                        
                        # 更新元数据
                        self.metadata[abs_path] = {
                            'mtime': file_path.stat().st_mtime
                        }
                        
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        # 删除不存在的文件
        current_relative_paths = {
            str(Path(file_path).relative_to(directory)) 
            for file_path in current_files
        }
        
        new_documents = []
        new_embeddings = []
        removed_count = 0
        
        for i, doc in enumerate(self.documents):
            doc_path = doc.split('\n\n')[0].replace('文件: ', '')
            if doc_path in current_relative_paths:
                new_documents.append(doc)
                new_embeddings.append(self.embeddings[i])
            else:
                removed_count += 1
                print(f"已删除文件: {doc_path}")
        
        if removed_count > 0:
            self.documents = new_documents
            self.embeddings = new_embeddings
            # 清理元数据
            new_metadata = {}
            for abs_path, meta in self.metadata.items():
                try:
                    relative_path = str(Path(abs_path).relative_to(directory))
                    if relative_path in current_relative_paths:
                        new_metadata[abs_path] = meta
                except ValueError:
                    continue
            self.metadata = new_metadata
            print(f"共删除了 {removed_count} 个文件")
        
        # 保存所有更新
        self.save_data()
        self.save_metadata()

    def clear_knowledge_base(self):
        """清空知识库"""
        self.documents = []
        self.embeddings = []
        self.metadata = {}
        
        # 删除存储文件
        if self.documents_file.exists():
            self.documents_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        print("知识库已清空") 