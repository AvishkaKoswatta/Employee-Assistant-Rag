#type: ignore
# chain.py
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from retriever import retriever 
import ollama

# -----------------------------
# Step 1: Prompt Template
# -----------------------------
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

# -----------------------------
# Step 2: Ollama LLM Runnable
# -----------------------------

# chain.py
class OllamaLLM(Runnable):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def invoke(self, inputs, *args, **kwargs):
        # Convert LangChain StringPromptValue to string if needed
        if hasattr(inputs, "to_string"):
            inputs = inputs.to_string()
        elif hasattr(inputs, "text"):  # some versions
            inputs = str(inputs.text)
        else:
            inputs = str(inputs)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": inputs}]
        )
        return response["message"]["content"]

ollama_llm = OllamaLLM(model_name="llama2")

# -----------------------------
# Step 3: Build the QA chain
# -----------------------------
qa_chain = prompt | ollama_llm

print("QA Chain ready using local Ollama model.")

# -----------------------------
# Step 4: Example usage
# -----------------------------
def answer_question(question_text):
    # Step 4a: Retrieve relevant context chunks from your retriever
    # retriever should return a string containing top-k relevant passages
    context = retriever(question_text)  # implement this in retriever.py

    # Step 4b: Generate the prompt with LangChain PromptTemplate
    final_prompt = prompt.format(context=context, question=question_text)

    # Step 4c: Invoke Ollama LLM
    answer = ollama_llm.invoke(final_prompt)
    return answer

# Example run
if __name__ == "__main__":
    question = "How many days of leave do employees get per year?"
    answer = answer_question(question)
    print("Question:", question)
    print("Answer:", answer)