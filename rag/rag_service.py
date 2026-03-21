"""
RAG文档合并
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from vector_store import VectorStoreService
from utils.prompt_loader import get_rag_prompt
from model.factory import chat_model


def print_prompt(prompt):
    print(prompt.to_string())
    return prompt


class RagSummarizeService():
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = get_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.init_chain()
        
        
    def init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    
    def ragSummarize(self,user_input: str):
        
        retriever_docs: list[Document] =  self.retriever.invoke(user_input)
        ret_str = ""
        count = 0
        for doc in retriever_docs:
            count += 1
            ret_str += f"【资料片段{count}】:\n内容：{doc.page_content}元数据：{doc.metadata}\n"
        
        return self.chain.invoke({
            "input": user_input,
            "context": ret_str
        })


if __name__ == '__main__':
    rag =  RagSummarizeService()
    res = rag.ragSummarize("家用扫地机器人有哪些")
    print(res)