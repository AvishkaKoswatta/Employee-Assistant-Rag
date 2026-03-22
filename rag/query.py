# # query.py
# from chain import qa_chain
# from retriever import retriever

# query = "What is the procedure for reporting harassment?"

# answer = qa_chain.invoke(query)
# print("Answer:")
# print(answer)

# source_docs = retriever.get_relevant_documents(query)
# print("\nSources:")
# for doc in source_docs:
#     title = doc.metadata.get("title", "Unknown")
#     section = doc.metadata.get("section", "N/A")
#     print(f"- {title} (Section: {section})")



from chain import qa_chain
from retriever import retriever

def ask_question(query: str):
    # Step 1: retrieve relevant documents
    docs = retriever.invoke(str(query))

    # Step 2: safe extraction
    context_list = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            context_list.append(doc.page_content)
        elif isinstance(doc, dict) and "text" in doc:
            context_list.append(doc["text"])

    context_text = "\n".join(context_list)

    # Step 3: run local Ollama LLM
    response = qa_chain.invoke({
        "context": context_text,
        "question": query
    })

    print("\nAnswer:\n", response.strip())

if __name__ == "__main__":
    while True:
        query = input("Enter your question (type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        ask_question(query)