import os
import sys
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

def generate_image_from_text(prompt: str) -> np.ndarray:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"]
        )
    )

    if not response.candidates or not response.candidates[0].content.parts:
        raise RuntimeError("No content in API response")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_bytes = part.inline_data.data
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            out_path = os.path.join(project_root, "input", "generated.png")
            cv2.imwrite(out_path, img)
            print(f"Image saved to {out_path}")
            return img

    raise RuntimeError("No image was returned by Gemini API")

if __name__ == "__main__":
    test_prompt = "a portrait of a young man, pencil sketch style, simple background"
    generated = generate_image_from_text(test_prompt)
    print(f"Shape: {generated.shape}, dtype: {generated.dtype}")