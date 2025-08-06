simple_prompt = '''
You are an assistant for question-answering tasks. Answer the question. If you don't know the answer, just say that you don't know. Use exactly three sentences and keep the answer concise.
Question: {question}
Answer:
'''

rag_prompt = '''
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use exactly three sentences and keep the answer concise.
Question: {question}
Context: {context}
Answer:
'''

judge_system_prompt = '''
You will be given a question and two different answers to that question. Based on your knowledge, judge which answer is better, in terms of factual accuracy and relevance to the question.

- Return "Answer A" if Answer A is better than Answer B, in terms of factual accuracy, answer quality, and relevance to the question.
- Return "Answer B" if Answer B is better than Answer A, in terms of factual accuracy, answer quality, and relevance to the question.
- Return "Both" if either:
    - There isn't a significant difference in the quality of Answer A vs Answer B.
    - You don't know whether to choose Answer A or Answer B.
'''

judge_human_prompt = '''
Question: {question}
Answer A: {answerA}
Answer B: {answerB}
'''