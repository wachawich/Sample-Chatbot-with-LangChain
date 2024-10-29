import pandas as pd
import numpy as np
import os

# Langchain
import key_param
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = key_param.OPENAI_API_KEY

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.1,
    max_tokens=2048,
    timeout=None,
)

template = """
    
    You are an AI expert specialized in artificial intelligence. 
    Provide clear and concise answers on topics like AI functionality, 
    Machine Learning, Deep Learning, applications of AI, and future trends. 
    Share useful and up-to-date information, suitable for readers at any knowledge level. 
    Use straightforward language, offer examples when appropriate, and ensure each response is helpful and relevant.

    this is question : {question}

    """

prompt = PromptTemplate(template=template)


chain = prompt | llm


def QA_chain(question, chain=chain):
    """
    QA chat function
    """

    fianl_answer = chain.invoke(
        {
            "question": {question},
        }
    )

    return fianl_answer.content



