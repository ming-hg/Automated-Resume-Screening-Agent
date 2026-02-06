import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
# 原有核心依赖（无需修改）
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
# 通义千问官方SDK（无需修改）
import dashscope
from dashscope import TextEmbedding

# 🔥 新增：导入ChatDeepSeek大模型+Prompt模板+输出解析器
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 🔥 新增：LangChain链（拼接检索+大模型，实现一键调用）
from langchain_core.runnables import RunnablePassthrough

# 加载.env所有环境变量（同时读取通义千问+DeepSeek的KEY）
load_dotenv()
# 配置通义千问嵌入SDK（无需修改）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
# 🔥 新增：获取DeepSeek API-KEY（后续模型自动读取）
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

# 原有自定义通义千问嵌入类（无需任何修改，直接复用）
class CustomTongyiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-v1"):
        self.model_name = model_name
        if not dashscope.api_key:
            raise ValueError("未配置DASHSCOPE_API_KEY！请在.env文件中添加")

    def embed_query(self, text: str) -> list[float]:
        text = text.strip().replace("\n", " ")
        if not text:
            return [0.0] * 1536
        response = TextEmbedding.call(model=self.model_name, input=[text])
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"通义千问嵌入失败：{response.message}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t.strip().replace("\n", " ") or "" for t in texts]
        response = TextEmbedding.call(model=self.model_name, input=texts)
        if response.status_code == 200:
            return [emb['embedding'] for emb in response.output['embeddings']]
        else:
            raise Exception(f"通义千问批量嵌入失败：{response.message}")

# 🔥 新增：构建RAG专属Prompt模板（核心，让大模型基于上下文回答）
# 模板规则：指定角色+限定上下文+用户问题，避免大模型幻觉
# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是一个专业的问答助手，仅根据提供的PDF文档上下文内容回答用户问题，
#     严格遵循以下规则：
#     1. 答案必须完全来自上下文，不得添加任何自己的知识；
#     2. 如果上下文没有相关信息，直接回答「未在PDF文档中找到相关答案」；
#     3. 答案简洁明了，贴合问题，不要冗余内容。"""),
#     ("human", "上下文：{context}\n\n用户问题：{question}")
# ])
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的前端专家，请根据提供的PDF文档上下文内容生成一份定制的面试题用于面试"""),
    ("human", "上下文：{context}\n\n用户问题：{question}")
])

# 完整RAG主逻辑（新增大模型调用，其余不变）
def main():
    # 1. 加载PDF（无需修改）
    pdf_path = os.path.abspath("E:\\ilearning\\AIProject\\langchainJS\\data\\zhangsan.pdf")
    # 兜底：若绝对路径报错，替换为相对路径（需项目根目录有data文件夹）
    # pdf_path = os.path.abspath("data/zhangsan.pdf")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ PDF加载成功，共{len(docs)}页")

    # 2. 文本分块（无需修改）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✅ PDF分块完成，总块数：{len(all_splits)}")

    # 3. 初始化自定义通义千问嵌入（无需修改）
    embeddings = CustomTongyiEmbeddings()

    # 4. FAISS向量库构建（无需修改）
    vector_store = FAISS.from_documents(all_splits, embeddings)
    print(f"✅ FAISS向量库构建完成，已添加{len(all_splits)}个分块")

    # 🔥 新增：构建检索器（指定返回Top3相关结果，平衡精准度和效率）
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 🔥 新增：初始化ChatDeepSeek大模型（默认使用deepseek-chat，适配问答）
    llm = ChatDeepSeek(
        model="deepseek-chat",  # 模型版本，无需修改
        temperature=0.1,       # 温度值，0.1适合精准问答，越小答案越确定
        max_tokens=1024        # 单次生成最大令牌数，足够日常问答
    )

    # 🔥 新增：构建完整RAG链（检索→拼接上下文→Prompt→大模型→解析输出）
    # 链式调用：用户问题→检索器获取上下文→格式化Prompt→大模型生成→解析为字符串
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 🔥 核心：用户问题（可自行修改测试）
    user_question = "输出一份面试题"

    # 🔥 运行RAG链，生成答案
    print(f"\n🔍 正在检索并回答问题：{user_question}")
    answer = rag_chain.invoke(user_question)

    # # 输出结果
    # print(f"\n✅ 最终答案：\n{answer}")
    #
    # # 可选：输出检索到的原始上下文（便于调试）
    # print(f"\n📄 检索到的相关PDF上下文（Top3）：")
    # contexts = retriever.invoke(user_question)
    # for i, doc in enumerate(contexts, 1):
    #     print(f"\n【第{i}条】页码：{doc.metadata['page']} | 内容：{doc.page_content[:150]}...")
    # ==============================================
    # 🔥 新增：面试题保存为TXT文件核心代码（直接复制粘贴）
    # ==============================================
    # 1. 创建输出文件夹（agent.py同级的output文件夹，不存在则自动创建）
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)  # exist_ok=True：文件夹已存在则不报错

    # 2. 生成唯一文件名（前端面试题_20260202_153020.txt，避免重复覆盖）
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"前端面试题_{current_time}.txt"
    file_path = output_dir / file_name

    # 3. 整合保存内容（排版清晰，含生成信息+面试题+检索上下文）
    save_content = f"""# 前端定制面试题（基于PDF自动生成）
    生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    PDF文件：{pdf_path}
    PDF总页数：{len(docs)}
    PDF总分块数：{len(all_splits)}
    检索相关分块数：3

    ==============================================
    ✅ 生成的面试题
    ==============================================
    {answer}

    ==============================================
    📄 检索的PDF相关上下文（面试题贴合依据）
    ==============================================
    """
    # 拼接检索上下文到保存内容
    contexts = retriever.invoke(user_question)
    for i, doc in enumerate(contexts, 1):
        save_content += f"\n【第{i}条】页码：{doc.metadata['page']} | 内容：{doc.page_content[:300]}...\n"

    # 4. 写入文件（utf-8编码避免中文乱码，w模式新建/覆盖文件）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(save_content)

    # 5. 打印保存成功提示（告知用户文件路径）
    print(f"\n📁 面试题已成功保存为文件：\n{file_path.absolute()}")
    # ==============================================
    # 🔥 保存代码结束
    # ==============================================

    # 原有代码：终端输出结果（可保留，也可删除，不影响保存）
    print(f"\n✅ 最终答案：\n{answer}")
    print(f"\n📄 检索到的相关PDF上下文（Top3）：")
    contexts = retriever.invoke(user_question)
    for i, doc in enumerate(contexts, 1):
        print(f"\n【第{i}条】页码：{doc.metadata['page']} | 内容：{doc.page_content[:150]}...")

# 执行主函数，捕获所有异常
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = str(e)
        # 针对性错误提示，便于排查
        if "DEEPSEEK_API_KEY" in error_msg or "deepseek" in error_msg.lower():
            print(f"❌ 错误：DeepSeek API-KEY未配置/无效，请检查.env中的DEEPSEEK_API_KEY")
        elif "DASHSCOPE_API_KEY" in error_msg:
            print(f"❌ 错误：通义千问API-KEY未配置/无效，请检查.env中的DASHSCOPE_API_KEY")
        elif "No such file or directory" in error_msg:
            print(f"❌ 错误：PDF文件不存在，请检查路径是否正确：{pdf_path}")
        else:
            print(f"❌ 程序运行异常：{error_msg}")