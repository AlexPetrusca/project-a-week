from typing import Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from prompts import simple_prompt, rag_prompt, judge_system_prompt, judge_human_prompt


def build_faiss_vectorstore():
    # get full documents
    loader = DirectoryLoader("./", glob="**/*.md", show_progress=True, use_multithreading=True)
    full_documents = loader.load()

    # chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    documents = text_splitter.split_documents(full_documents)

    # store in FAISS with nomic embeddings (ollama)
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
    vectorstore.save_local("rsc/faiss_index")

    return vectorstore


def first_chain():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    try:
        vectorstore = FAISS.load_local("rsc/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    except RuntimeError:
        vectorstore = build_faiss_vectorstore()

    return (
        {
            "context": vectorstore.as_retriever(),
            "question": RunnablePassthrough(),
        }
        | PromptTemplate.from_template(rag_prompt)
        | OllamaLLM(model="llama3.2:1b")
        | StrOutputParser()
)


def second_chain():
    return (
            PromptTemplate.from_template(simple_prompt)
            | OllamaLLM(model="llama3.2:1b")
            | StrOutputParser()
    )


def judge_chain():
    class Judgement(BaseModel):
        choice: Literal["Answer A", "Answer B", "Both"] = Field(
            ...,
            description="Given a question and two different answers to that question, select whether Answer A is better, Answer B is better, or both are about the same.",
        )

    judge_llm = ChatOllama(model="gemma3n")
    structured_judge_llm = judge_llm.with_structured_output(Judgement)

    return (
            ChatPromptTemplate.from_messages(
                [
                    ("system", judge_system_prompt),
                    ("human", judge_human_prompt),
                ]
            )
            | structured_judge_llm
    )


load_dotenv()  # load environment variables

modelA = first_chain()
modelB = second_chain()
judge = judge_chain()

queries =[
    # Expect better performance
    "What is the seam carving algorithm?",
    "What is the relationship between a prefix sum and an integral image?",
    "What are some of the best ways to represent a graph in code?",
    # Expect about equal performance
    "Given an integer array nums, handle multiple queries of the following types: 1. Update the value of an element in nums. 2. Calculate the sum of the elements of nums between indices left and right inclusive where left <= right. Give a broad strokes overview of how you'd solve this?",
    "What data structures and algorithms do you use to solve the trapping rainwater coding problem?",
    "What is a multiset good for?",
    # Expect worse performance
    "What is a fenwick tree?",
]

RUNS_PER_QUERY = 25
for query in queries:
    print("---------")
    print(query)
    print("---------")

    grade = 0
    for i in range(RUNS_PER_QUERY):
        answerA = modelA.invoke(query)
        # print(answerA)
        # print("\n---------\n")

        answerB = modelB.invoke(query)
        # print(answerB)
        # print("\n---------\n")

        judge_response = judge.invoke({
            "question": query,
            "answerA": answerA,
            "answerB": answerB,
        })
        print(judge_response.choice)

        if judge_response.choice == 'Answer A':
            grade += 1
        elif judge_response.choice == 'Both':
            grade += 0.5

    print(f"Grade: {100 * grade / RUNS_PER_QUERY}%")