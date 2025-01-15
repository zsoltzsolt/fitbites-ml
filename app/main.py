import os
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from app.services.nutrition import calculate_meal_nutrition, retrieve_similar_ingredients
from app.settings import settings
from app.services.vectorstore import retriever
import pandas as pd
from typing import AsyncGenerator, NoReturn
from dotenv import load_dotenv
from openai import AsyncOpenAI
import uuid

load_dotenv()

app = FastAPI()

client = AsyncOpenAI()

async def get_ai_response(message: str) -> AsyncGenerator[str, None]:
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly knowledgeable and specialized virtual assistant acting as a professional nutritionist. "
                    "Your sole purpose is to provide accurate, evidence-based, and easy-to-understand answers related to nutrition, dietary habits, healthy eating, meal planning, calorie intake, macronutrient breakdowns, vitamins, minerals, hydration, and food-related health concerns. "
                    "You are expected to address a wide range of nutrition-related questions, including topics like weight management, nutritional requirements, food allergies, meal preparation tips, and how certain foods impact health. "
                    "Your tone should always be professional, supportive, and encouraging, helping users make informed decisions about their diet and health."
                    "\n\n"
                    "You are strictly prohibited from answering questions or engaging in discussions that are unrelated to nutrition. "
                    "If the user asks something outside your domain—such as questions about technology, history, mathematics, or any other unrelated field—respond politely and clearly explain that your expertise is limited to nutrition and dietary topics. "
                    "For example, you might say, 'I'm here to help with nutrition-related questions. Unfortunately, I can't provide information on that topic.' "
                    "Your refusal should always be polite and maintain a professional tone while redirecting the user back to nutrition-related inquiries where possible."
                    "\n\n"
                    "If the user asks unclear or vague questions, politely request clarification or provide options for related topics you can discuss. "
                    "Your goal is to create a positive experience by providing focused and accurate guidance within your domain while remaining strictly specialized in nutrition. "
                    "You should never attempt to answer questions outside your expertise, even if the user insists."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        stream=True,
    )

    all_content = ""
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            all_content += content
            yield all_content



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> NoReturn:
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        uid = uuid.uuid4()
        print(str(uid))
        async for text in get_ai_response(message):
            print(text)
            await websocket.send_text(str(uid) + " " + text)


def save_temp_file(upload_file: UploadFile, destination_path: str):
    with open(destination_path, "wb") as buffer:
        buffer.write(upload_file.file.read())

def remove_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
       
@app.get("/search")
async def search_ingredients(query: str = Query(..., description="Search query for similar ingredients")):
    try:
        print(query)
        similar_ingredients = retrieve_similar_ingredients(query, 5)
        print(similar_ingredients)
        if not similar_ingredients:
            raise HTTPException(status_code=404, detail="No similar ingredients found")
        return {"ingredients":similar_ingredients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    temp_file_path = os.path.join(settings.CURRENT_FOLDER, file.filename)
    save_temp_file(file, temp_file_path)  

    try:
        with open(temp_file_path, "rb") as image_file:
            meal_nutrition = calculate_meal_nutrition(image_file)
        if not meal_nutrition:
            return JSONResponse({"error": "Failed to extract ingredients or calculate nutrition."}, status_code=400)

        return JSONResponse(meal_nutrition)
    finally:
        print("done")
