from dotenv import load_dotenv

load_dotenv()

from adaptive_rag.graph.graph import app

if __name__ == "__main__":
    print("Hello Adaptive RAG")
    print(app.invoke(input={"question": "agent memory"}))
