from google import genai
from google.genai import types
import llm_keys
from pydantic import BaseModel

class SimpleResponse(BaseModel):
    response: str

def call_gemini(
    gemini_client: genai.Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: BaseModel = SimpleResponse,
    thinking_budget: int = 512,
    verbose: bool = False
):

    response = gemini_client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
        contents=user_prompt,
    )

    if verbose:
        print(f"[{model_name}] Response raw:\n", response.text)

    return response_schema.model_validate_json(response.text)


if __name__ == "__main__":
    call_gemini(
        gemini_client=genai.Client(api_key=llm_keys.GEMINI_KEY),
        model_name="gemini-2.5-flash-lite-preview-06-17",
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        response_schema=SimpleResponse,
        thinking_budget=512,
        verbose=True
    )