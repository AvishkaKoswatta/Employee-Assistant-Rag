# # query.py
# from chain import qa_chain
# from retriever import retriever
# from query_enhancer import enhance_query

# def ask_question(query: str):
#     query = enhance_query(query)
#     docs = retriever.invoke(str(query))

    
#     context_list = []
#     for doc in docs:
#         if hasattr(doc, "page_content"):
#             context_list.append(doc.page_content)
#         elif isinstance(doc, dict) and "text" in doc:
#             context_list.append(doc["text"])

#     context_text = "\n".join(context_list)

#     response = qa_chain.invoke({
#         "context": context_text,
#         "question": query
#     })

#     print("\nAnswer:\n", response.strip())

# if __name__ == "__main__":
#     while True:
#         query = input("Enter your question (type 'exit' to quit): ")
#         if query.lower() in ["exit", "quit"]:
#             break
#         ask_question(query)







# query.py
from chain import answer_question

def ask_question(query: str):
    answer = answer_question(query)
    print("\nAnswer:\n", answer.strip())

if __name__ == "__main__":
    while True:
        query = input("Enter your question (type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        ask_question(query)