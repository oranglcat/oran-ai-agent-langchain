import os.path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import listdir_with_allowed_type, pdf_loader, txt_loader,get_file_md5_hex
from utils.logger_handler import logger

"""
向量数据库服务类，存入文档片段，生成检索器
"""


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            embedding_function=embed_model,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_documents(self):
        def check_md5(md5_str):
            # 判断md5文件是否存在,如果不存在则创建文件
            if not os.path.exists(get_abs_path(chroma_conf["md5_hextext"])):
                open(get_abs_path(chroma_conf["md5_hextext"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hextext"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if md5_str == line.strip():
                        return True
                return False

        def save_md5(md5_str: str):
            with open(get_abs_path(chroma_conf["md5_hextext"]), "a", encoding="utf-8") as f:
                f.write(md5_str + "\n")

        def get_file_documents(file_path: str):
            if file_path.endswith("txt"):
                return txt_loader(file_path)
            if file_path.endswith("pdf"):
                return pdf_loader(file_path)
            return []

        allowed_document_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for file_path in allowed_document_path:
            #获取文件的md5值
            md5_hex = get_file_md5_hex(file_path)
            if check_md5(md5_hex):
                logger.error(f"[加载知识库]{file_path}文件已存在")
                continue
            try:
                documents: list[Document] = get_file_documents(file_path)
                if not documents:
                    logger.warning(f"[加载知识库]{file_path}没有有效文件")
                    continue

                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库]{file_path}分片后没有有效片段")
                    continue

                self.vector_store.add_documents(split_documents)
                save_md5(md5_hex)
                logger.info(f"[加载知识库]{file_path}文件成功")

            except Exception as e:
                logger.error(f"[加载知识库]{file_path}失败，{str(e)}", exc_info= True)
                continue

if __name__ == '__main__':
    vector_store = VectorStoreService()
    retriever =  vector_store.get_retriever()
    vector_store.load_documents()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)