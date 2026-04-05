from rag_groq import run_rag


if __name__ == "__main__":
    query = "What is the effectiveness of ribavirin for treating Lassa fever?"

    result = run_rag(query, k=6)

    print("\nFINAL ANSWER:\n")
    for line in result:
        print("-", line)
