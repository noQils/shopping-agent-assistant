# AI Shopping Assistant

AI Shopping Assistant is a Streamlit application that demonstrates a catalog-aware shopping agent. Users can ask natural-language shopping questions, compare products by constraints such as price or rating, upload product images for visual matching, and complete a simulated checkout against a local SQLite store database.

The project is designed as a practical example of an agentic retail assistant: the language model does not invent catalog data. It must use local tools for product search, semantic retrieval, image interpretation, ratings, and checkout before making recommendations or placing orders.

## Demo

https://github.com/user-attachments/assets/cb6b7c9e-d950-45f1-a169-84bcf8186006

## Features

- Chat-based shopping assistant built with Streamlit.
- Local product catalog with 100 seeded products across grocery, household, personal-care, and pantry categories.
- Keyword search over product name, category, and description.
- Hybrid semantic search using `sentence-transformers` embeddings plus full-text search.
- Product filters for explicit user constraints such as maximum price, organic preference, and minimum rating.
- Image upload flow for product matching using a vision model.
- Ratings lookup from local review data.
- Simulated checkout that writes orders and order items to SQLite.
- Session-based chat memory using an in-memory LangGraph checkpointer.
- Optional debug logging through the `DEBUG=true` environment variable.

## How It Works

The application starts in [app.py](app.py). On launch, it ensures that the local database exists, initializes the Streamlit interface, and routes user messages to the shopping agent.

The agent is defined in [shopping_agent_assistant/agent.py](shopping_agent_assistant/agent.py). It uses a strict system prompt that instructs the model to ground all recommendations in tool results. The primary model is Gemini 2.5 Flash through `langchain-google-genai`; if initialization fails, the app falls back to a Groq-hosted model. Image understanding uses Groq's vision-capable model.

The local database is created by [shopping_agent_assistant/db/setup_db.py](shopping_agent_assistant/db/setup_db.py). It seeds products, reviews, full-text search data, product embeddings, and order tables into `data/store.db`.

The agent has access to these tools:

- `search_product`: keyword-style lookup with optional hard filters.
- `semantic_search_product`: hybrid semantic retrieval using FTS5 and vector similarity.
- `describe_product_image`: extracts product attributes from an uploaded image.
- `get_product_rating`: returns average rating and review count for a product.
- `checkout`: creates a simulated order for selected product IDs.

## Project Structure

```text
shopping-agent-assistant/
├── app.py
├── README.md
├── demo.mp4
├── pyproject.toml
├── resources/
│   ├── car.png
│   ├── elephant.png
│   ├── honey.png
│   ├── oats.png
│   ├── paper towel.png
│   ├── pasta.png
│   └── toothpaste.png
├── data/
│   └── store.db
└── shopping_agent_assistant/
    ├── agent.py
    ├── db/
    │   └── setup_db.py
    ├── runtime/
    │   └── runtime_log.py
    └── tools/
        ├── checkout.py
        ├── describe_product_image.py
        ├── get_product_rating.py
        ├── search_product.py
        └── semantic_search_product.py
```

`data/store.db` is generated automatically when the app runs and is intentionally ignored by Git.

## Requirements

- Python 3.14 or newer, as declared in `pyproject.toml`.
- A Groq API key for the fallback chat model and image analysis.
- Optionally, a Google Gemini API key for the primary chat model.

Environment variables are loaded from `.env`.

## Setup

1. Clone the repository and enter the project directory.

```bash
git clone https://github.com/noQils/shopping-agent-assistant.git
cd shopping-agent-assistant
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

4. Create a `.env` file.

```bash
cp .env.example .env
```

Then set the API keys you want to use:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEBUG=false
```

`GROQ_API_KEY` is required for image-based product matching. `GOOGLE_API_KEY` is optional if you want the main chat agent to use Gemini.

## Running the App

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser to use the shopping assistant.

If the database does not exist yet, the app creates `data/store.db` automatically on startup. The first run may take longer because product embeddings are generated with `sentence-transformers`.

## Example Prompts

Try prompts like:

```text
Find organic honey under $20 with at least 4.5 stars.
Compare the best pasta options under $5.
Recommend a high-rated pantry staple for meal prep.
I need household basics under $10 with good reviews.
Buy the best matching product.
```

For image search, use the sidebar uploader, choose a product image, and click **Find similar products**. The assistant will analyze the uploaded image, search the catalog, and recommend matching or closest available products.

## Data Model

The local SQLite database contains:

- `products`: catalog items with name, category, price, description, and organic flag.
- `products_fts`: FTS5 virtual table for keyword retrieval.
- `product_embeddings`: serialized product vectors for semantic matching.
- `reviews`: product-level ratings and review text.
- `orders`: simulated checkout orders.
- `order_items`: products attached to each simulated order.

This makes the project self-contained: product recommendations, ratings, and checkout records all come from local data rather than an external commerce API.

## Development Notes

- Set `DEBUG=true` in `.env` to print tool calls, search parameters, tool results, and assistant responses in the terminal.
- Delete `data/store.db` if you want to rebuild the seeded database from scratch.
- The app keeps chat state in Streamlit session state and uses a generated thread ID per conversation.
- The checkout flow is intentionally simulated. It records orders locally and returns a confirmation message, but it does not process payments or contact a fulfillment service.

## Limitations

- The catalog is seeded sample data, not a live inventory system.
- Image matching depends on the vision model's description of the uploaded file and the products available in the local catalog.
- The in-memory checkpointer does not persist conversation memory after the Python process stops.
- Checkout is for demonstration only.

## Tech Stack

- Streamlit for the web interface.
- LangChain agent tooling for model/tool orchestration.
- LangGraph in-memory checkpointing for conversation continuity.
- SQLite for local catalog, review, and order storage.
- SQLite FTS5 for keyword search.
- Sentence Transformers for local product embeddings.
- Groq and Google Gemini model integrations through LangChain.
