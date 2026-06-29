import os
import base64
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with catalog search tools.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract catalog-search attributes. "
                "Prefer broad retail catalog terms over overly specific guesses when the image is ambiguous. "
                "For example, a white perforated absorbent roll may be paper towels even if it resembles toilet paper. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: the most likely product type\n"
                "- possible_product_types: 2-4 plausible product types, ordered by likelihood\n"
                "- search_query: the best short catalog keyword to search first\n"
                "- fallback_search_queries: 2-4 alternative catalog keywords to try if the first search has no exact match\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- visual_attributes: short phrases describing visible shape, material, packaging, texture, and use case\n"
                "- description: one sentence describing the product\n"
                "Use singular/plural variants where helpful, such as 'paper towels' and 'paper towel'."
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content
