# DataDendron: LLM-Powered Hierarchical Item Categorizer with Refinement

This Python script demonstrates an advanced technique for automatically assigning hierarchical categories to structured text items (like research notes, article summaries, product feedback, etc.) using Large Language Models (LLMs) via LangChain.

A key feature is the **iterative refinement mechanism**: If an initial category assigned by the LLM contains too many items (exceeding a defined threshold), the script automatically re-prompts the LLM to subdivide that specific category further, promoting a more granular and balanced hierarchical structure.

This project showcases several important techniques for building robust and sophisticated LLM applications.

## Key Features & Techniques Showcased

*   **Hierarchical Categorization:** Uses an LLM (configurable, e.g., OpenAI's GPT series) to assign multi-level category paths (e.g., `["Topic A", "Subtopic 1", "Detail 1a"]`) based on item content.
*   **Iterative Category Refinement:** Automatically identifies categories exceeding a predefined size threshold (`MAX_ITEMS_PER_CATEGORY_TARGET`) and invokes the LLM again with a specialized prompt to subdivide only those oversized categories.
*   **Structured Output w/ Pydantic:** Defines the expected LLM output structure using Pydantic models (`CategorizationList`, `CategorizationItem`), ensuring type safety and clear data contracts.
*   **Robust LLM Parsing:** Employs LangChain's `PydanticOutputParser` and potentially `OutputFixingParser` (with fallback) to reliably parse potentially malformed JSON output from the LLM, automatically attempting to fix errors.
*   **API Rate Limit Handling:** Implements automatic exponential backoff retries using the `tenacity` library specifically for `openai.RateLimitError`, making the script more resilient to temporary API issues.
*   **Modularity & Configuration:** Organizes code into a class (`HierarchicalCategorizer`) with helper functions. Key parameters (model name, thresholds, temperature) are configurable via constants.
*   **Sanitization:** Includes helper functions for sanitizing category names returned by the LLM for consistency.
*   **Standalone & Demonstrative:** Provided as a single, runnable script with inline example data for easy demonstration and understanding.

## How It Works

1.  **Initialization:** Sets up the LangChain components: LLM (`ChatOpenAI`), Pydantic output parsers, prompt templates, and runnable chains.
2.  **Initial Categorization:**
    *   The input `DataCollection` (list of `DataItem` objects) is processed in chunks (`MAX_ITEMS_PER_CHUNK_CAT`) to avoid overwhelming the LLM context window.
    *   For each chunk, the `categorization_chain` is invoked. The LLM assigns an initial hierarchical category path to each item based on its headline.
    *   Results are parsed using the robust `fixing_parser`, and category paths are updated in a central map.
3.  **Refinement Loop:**
    *   After initial categorization, the script checks the distribution of items across the generated category paths.
    *   Any category path (that is not too deep, e.g., < 3 levels) containing more items than `MAX_ITEMS_PER_CATEGORY_TARGET` is identified.
    *   For each identified oversized category, the `refinement_chain` is invoked with the items belonging to that category. The LLM is prompted to create one additional, more specific subcategory level for those items.
    *   The category paths for the affected items are updated in the central map with the deeper, refined paths provided by the LLM.
    *   This refinement process iterates up to `MAX_REFINEMENT_ITERATIONS` times or until no categories exceed the target threshold.
4.  **Output:** Returns a new `DataCollection` object where items have their final (potentially refined) `category_path` assigned. The script's example usage also prints the final category assignments and a tree structure representation.

## Requirements

*   Python 3.9+
*   Required Python packages:
    ```bash
    pip install langchain langchain-openai python-dotenv pydantic tenacity openai
    ```

## Setup

1.  **Clone the repository or download the script.**
2.  **Install dependencies:**
    ```bash
    pip install langchain langchain-openai python-dotenv pydantic tenacity openai
    ```
3.  **Create `.env` file:** Create a file named `.env` in the same directory as the script.
4.  **Add API Key:** Add your OpenAI API key to the `.env` file:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    **Important:** Ensure the `.env` file is added to your `.gitignore` file to prevent accidentally committing your API key.

## Running the Example

The script includes a self-contained example in the `if __name__ == "__main__":` block. To run it:

```bash
python standalone_hierarchical_categorizer.py
```

This will:
1.  Initialize the `HierarchicalCategorizer`.
2.  Create sample `DataItem` objects.
3.  Run the categorization and refinement process (making actual calls to the OpenAI API specified in your `.env` file).
4.  Print the progress, final category assignments for each item, and the resulting hierarchical category tree structure to the console.

## Code Structure Overview

*   **Configuration:** Constants for model names, thresholds, API key loading.
*   **Data Models:** Pydantic models (`CategoryPath`, `DataItem`, `DataCollection`) defining the data structure.
*   **Prompt Templates:** LangChain `PromptTemplate` definitions for categorization and refinement (`CATEGORIZATION_TEMPLATE`, `REFINEMENT_TEMPLATE`).
*   **Pydantic Output Models:** Pydantic models (`CategorizationItem`, `CategorizationList`) defining the expected LLM JSON output structure.
*   **Helper Functions:** Utility functions (`_log_retry_attempt`, `_sanitize_category_name`).
*   **`HierarchicalCategorizer` Class:** The main class containing the core logic for initialization, chunking, LLM invocation (with retry), parsing, and the refinement loop.
*   **Example Usage (`if __name__ == "__main__":`)**: Demonstrates how to instantiate the class and use it with sample data.

## Customization

*   **LLM Model:** Change `CATEGORIZATION_MODEL_NAME` to use a different OpenAI model (e.g., `gpt-4.1-mini`).
*   **Thresholds:** Adjust `MAX_ITEMS_PER_CHUNK_CAT` and `MAX_ITEMS_PER_CATEGORY_TARGET` to control chunking and refinement behavior.
*   **Prompts:** Modify `CATEGORIZATION_TEMPLATE` and `REFINEMENT_TEMPLATE` to adjust the categorization logic or desired category structure.
*   **Data Models:** Adapt the `DataItem` model if your input items have different fields.
