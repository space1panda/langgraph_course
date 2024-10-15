from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello C-RAG")
    print(app.invoke(input={"question": "What is agent memory?"}))
