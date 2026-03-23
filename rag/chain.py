# #type: ignore
# # chain.py
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import Runnable
# from retriever import retriever 
# import ollama

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# Use the following context from the Employee Handbook to answer the question.
# If the answer is not present, say "I don't know".

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# class OllamaLLM(Runnable):
#     def __init__(self, model_name="llama2"):
#         self.model_name = model_name

#     def invoke(self, inputs, *args, **kwargs):
        
#         if hasattr(inputs, "to_string"):
#             inputs = inputs.to_string()
#         elif hasattr(inputs, "text"): 
#             inputs = str(inputs.text)
#         else:
#             inputs = str(inputs)

#         response = ollama.chat(
#             model=self.model_name,
#             messages=[{"role": "user", "content": inputs}]
#         )
#         return response["message"]["content"]

# ollama_llm = OllamaLLM(model_name="llama2")


# qa_chain = prompt | ollama_llm

# print("QA Chain ready using local Ollama model.")

# def answer_question(question_text):
    
#     context = retriever(question_text) 

    
#     final_prompt = prompt.format(context=context, question=question_text)

#     answer = ollama_llm.invoke(final_prompt)
#     return answer


# if __name__ == "__main__":
#     question = "How many days of leave do employees get per year?"
#     answer = answer_question(question)
#     print("Question:", question)
#     print("Answer:", answer)




#type: ignore
# chain.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from .retriever import retriever 
import ollama
import json


# =========================
# ✅ MAIN QA PROMPT
# =========================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context from the Employee Handbook to answer the question.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================
# ✅ OLLAMA LLM (ANSWERING)
# =========================
class OllamaLLM(Runnable):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def invoke(self, inputs, *args, **kwargs):

        if hasattr(inputs, "to_string"):
            inputs = inputs.to_string()
        elif hasattr(inputs, "text"):
            inputs = str(inputs.text)
        else:
            inputs = str(inputs)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": inputs}]
        )

        return response["message"]["content"]


ollama_llm = OllamaLLM(model_name="llama2")

print("QA Chain ready using local Ollama model.")


# =========================
# ✅ QUERY REWRITE + METADATA (COMBINED)
# =========================
class OllamaRewriter(Runnable):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def invoke(self, inputs, *args, **kwargs):
        original_query = inputs["original_query"]

        rewrite_prompt = f"""
You are an AI assistant for a RAG system.

Your tasks:
1. Rewrite the query to improve retrieval
2. Extract the most relevant metadata filter

Return ONLY valid JSON in this format:
{{
  "query": "...",
  "filters": {{
    "section": "..."
  }}
}}

Rules:
- If no clear section → return "filters": {{}}
- Do NOT explain anything
- Only output JSON

Original query: {original_query}
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": rewrite_prompt}]
        )

        return response["message"]["content"].strip()


query_rewriter = OllamaRewriter(model_name="llama2")


# =========================
# ✅ NORMALIZE FILTERS
# =========================
def normalize_filters(filters: dict) -> dict:
    if "section" in filters:
        return {"section": filters["section"].title()}
    return {}


def rewrite_query(original_query: str):
    response = query_rewriter.invoke({"original_query": original_query})

    try:
        data = json.loads(response)
        query = data.get("query", original_query)
        filters = normalize_filters(data.get("filters", {}))
        return query, filters
    except:
        return original_query, {}


# =========================
# ✅ HyDE GENERATOR
# =========================
class OllamaHyDE(Runnable):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def invoke(self, inputs, *args, **kwargs):
        query = inputs["query"]

        hyde_prompt = f"""
You are an AI assistant generating a hypothetical answer for retrieval purposes.

Write a detailed, well-structured answer to the following question.
Do NOT say "I don't know". Assume relevant information exists.

Question: {query}

Hypothetical Answer:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": hyde_prompt}]
        )

        return response["message"]["content"].strip()


hyde_generator = OllamaHyDE(model_name="llama2")


def generate_hyde(query: str) -> str:
    return hyde_generator.invoke({"query": query})


# =========================
# ✅ FINAL PIPELINE
# =========================
def answer_question(question_text):

    # 🔥 Step 1: Rewrite + Extract Filters
    rewritten_query, filters = rewrite_query(question_text)

    # 🔥 Step 2: Generate HyDE
    hyde_doc = generate_hyde(rewritten_query)

    # 🔍 Debug
    print("\nOriginal Query:", question_text)
    print("Rewritten Query:", rewritten_query)
    print("Filters:", filters)
    print("HyDE Document:", hyde_doc[:300], "...")

    # 🔥 Step 3: Filtered Retrieval
     
    docs = retriever.invoke(hyde_doc)

    if filters.get("section"):
        docs = [doc for doc in docs if doc.metadata.get("section") == filters["section"]]

    context = "\n".join([doc.page_content for doc in docs])

    # 🔥 Step 4: Fallback
    if not context.strip():
        print("⚠️ No results with filter → fallback to full search")
        docs = retriever.invoke(hyde_doc)
        context = "\n".join([doc.page_content for doc in docs])

    # 🔥 Step 5: Final Answer
    final_prompt = prompt.format(
        context=context,
        question=question_text
    )

    answer = ollama_llm.invoke(final_prompt)

    return answer


# =========================
# ✅ TEST
# =========================
if __name__ == "__main__":
    question = "How many days of leave do employees get per year?"

    answer = answer_question(question)

    print("\nFinal Answer:\n", answer)