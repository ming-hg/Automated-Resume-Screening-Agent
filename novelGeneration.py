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

    # 单文本嵌入方法（无需修改，保持原有逻辑）
    def embed_query(self, text: str) -> list[float]:
        text = text.strip().replace("\n", " ")
        if not text:
            return [0.0] * 1536
        response = TextEmbedding.call(model=self.model_name, input=[text])
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"通义千问嵌入失败：{response.message}")

    # 🔥 核心修改：批量嵌入方法（添加分批次处理，每批最多25个）
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 1. 原有文本预处理：去除空格、换行，空文本替换为""
        texts = [t.strip().replace("\n", " ") or "" for t in texts]
        # 2. 定义通义千问接口最大批次大小（固定25，不可修改）
        BATCH_MAX_SIZE = 25
        # 3. 拆分文本列表为多个批次（每批≤25个）
        batches = [texts[i:i + BATCH_MAX_SIZE] for i in range(0, len(texts), BATCH_MAX_SIZE)]
        # 4. 初始化空列表，用于合并所有批次的嵌入结果
        all_embeddings = []

        # 5. 循环处理每个批次，调用通义千问接口
        for batch_num, batch_texts in enumerate(batches, 1):
            print(f"📦 处理嵌入批次 {batch_num}/{len(batches)}，本批文本数：{len(batch_texts)}")
            # 调用通义千问批量嵌入接口（单批次≤25，符合限制）
            response = TextEmbedding.call(model=self.model_name, input=batch_texts)
            # 6. 批次响应处理：成功则提取向量，失败则抛出原错误
            if response.status_code == 200:
                batch_embeddings = [emb['embedding'] for emb in response.output['embeddings']]
                all_embeddings.extend(batch_embeddings)  # 合并批次结果
            else:
                raise Exception(f"通义千问批量嵌入（批次{batch_num}）失败：{response.message}")

        # 7. 返回所有文本的完整嵌入向量（与原有方法返回格式一致，不影响后续逻辑）
        return all_embeddings

# 🔥 新增：构建RAG专属Prompt模板（核心，让大模型基于上下文回答）
# 模板规则：指定角色+限定上下文+用户问题，避免大模型幻觉
# 🔥 核心修改：大唐穿越脑洞小说生成Prompt（Agent的核心指令）
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是专业的历史脑洞小说作家，擅长大唐穿越题材，**逻辑合规是第一优先级**，严格遵循以下规则生成2000字/章的小说，缺一不可：
1. 【核心逻辑】所有内容完全贴合「大唐穿越脑洞小说构思表」的**人设/金手指/剧情大纲/6大逻辑约束模块**，不偏离、不新增、不矛盾，情节发展必须贴合构思表的阶段衔接节点；
2. 【字数要求】单章节正文纯汉字不少于2000字，通过**丰富的场景描写/人物对话/心理活动/细节动作**填充，不凑字数、不重复、不注水；
3. 【人设一致】主角/配角的性格、行为、语言完全贴合构思表设定，主角的金手指使用严格遵循「激活条件/限制/冷却」规则，杜绝万能金手指；
4. 【情节聚焦】每章围绕构思表指定的**核心剧情目标**展开，仅解决1个核心矛盾，不新增无关剧情/配角，情节推进按「铺垫→出现矛盾→尝试解决→小高潮→衔接下章」的逻辑链展开；
5. 【脑洞落地】现代元素与大唐的结合必须有落地过程，符合大唐人的认知，不直接照搬现代事物，避免违和；
6. 【历史底线】严格遵循构思表的历史时代逻辑底线，不出现正史硬伤，非核心脑洞不影响大唐整体历史走向；
7. 【前情呼应】若生成后续章节，必须**紧密呼应上一章的结尾/伏笔**，做到前后剧情连贯，无逻辑断层；
8. 【排版要求】分段落排版（每段3-5行），对话独立成段，2000字内容分布均匀，小高潮出现在章节后半段，结尾留轻微悬念（贴合构思表的衔接节点）。"""),
    ("human", "上下文：{context}\n\n用户问题：{question}")
])
# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是一个专业的前端专家，请根据提供的PDF文档上下文内容生成一份定制的小说用于面试"""),
#     ("human", "上下文：{context}\n\n用户问题：{question}")
# ])

# 完整RAG主逻辑（新增大模型调用，其余不变）
def main():
    # 1. 加载PDF（无需修改）
    pdf_path = os.path.abspath("E:\\AI-Project\\Automated-Resume-Screening-Agent\\data\\datang_novel_ider.pdf")
    # 兜底：若绝对路径报错，替换为相对路径（需项目根目录有data文件夹）
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # 🔥 新增：初始化ChatDeepSeek大模型（默认使用deepseek-chat，适配问答）
    llm = ChatDeepSeek(
        model="deepseek-chat",  # 模型版本，无需修改
        temperature=0.5,  # 核心：从0.6降至0.5（逻辑优先，保留适度脑洞）
        max_tokens=8192,  # 保持不变，给足2000字生成空间
        top_p=0.9,  # 新增：控制生成的多样性，0.9兼顾逻辑和细节
    )

    # 🔥 新增：构建完整RAG链（检索→拼接上下文→Prompt→大模型→解析输出）
    # 链式调用：用户问题→检索器获取上下文→格式化Prompt→大模型生成→解析为字符串
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    lv = 1
    # 选项1：生成单章（优先选这个，逐章生成更易保证2000字质量）
    user_question = f"请严格基于构思表，生成大唐穿越脑洞小说第{lv}章，单章节正文不少于2000字，细节丰富，分段落排版"

    # 🔥 运行RAG链，生成答案
    print(f"\n🔍 正在检索并回答问题：{user_question}")
    answer = rag_chain.invoke(user_question)

    # 1. 创建输出文件夹（agent.py同级的output文件夹，不存在则自动创建）
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)  # exist_ok=True：文件夹已存在则不报错

    # 2. 生成唯一文件名（生成小说_20260202_153020.txt，避免重复覆盖）
    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"datang_{lv}.txt"
    file_path = output_dir / file_name

    # 3. 整合保存内容（排版清晰，含生成信息+小说+检索上下文）
    save_content = f"""# 生成小说（基于PDF自动生成）
    生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    PDF文件：{pdf_path}
    PDF总页数：{len(docs)}
    PDF总分块数：{len(all_splits)}
    检索相关分块数：3

    ==============================================
    ✅ 生成的小说
    ==============================================
    {answer}

    ==============================================
    📄 检索的PDF相关上下文（小说贴合依据）
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
    print(f"\n📁 小说已成功保存为文件：\n{file_path.absolute()}")
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