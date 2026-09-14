from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)


context = """
Deep learning is a subfield of machine learning.

Neural networks are used in deep learning.
"""

question = "What is deep learning?"


final_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


print(final_prompt)