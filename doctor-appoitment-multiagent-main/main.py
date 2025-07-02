from fastapi import FastAPI
from pydantic import BaseModel
from agent import DoctorAppointmentAgent
from langchain_core.messages import HumanMessage
import os

os.environ.pop("SSL_CERT_FILE", None)


app = FastAPI()

# Define Pydantic model to accept request body
class UserQuery(BaseModel):
    id_number: int
    messages: str

agent = DoctorAppointmentAgent()

@app.post("/execute")
def execute_agent(user_input: UserQuery):
    app_graph = agent.workflow()
    
    # Prepare agent state as expected by the workflow
    input = [
        HumanMessage(content=user_input.messages)
    ]
    query_data = {
        "messages": input,
        "id_number": user_input.id_number,
        "next": "",
        "query": "",
        "current_reasoning": "",
    }
    #config = {"configurable": {"thread_id": "1", "recursion_limit": 100}}  

    response = app_graph.invoke(query_data,config={"recursion_limit": 20})
    return {"messages": response["messages"]}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6060, reload=True)








