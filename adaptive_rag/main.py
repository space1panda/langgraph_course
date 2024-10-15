from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Adaptive RAG")
    print(app.invoke(input={"question": "did Appolo mission really happened"}))
    # print(app.invoke(input={"question": "agent memory"}))
