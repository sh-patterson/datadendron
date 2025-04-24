# standalone_hierarchical_categorizer.py
"""
Standalone Hierarchical Item Categorizer using LangChain and LLMs.

This script demonstrates a technique for assigning hierarchical categories
to structured text items (e.g., summaries, notes, facts extracted from documents)
using Large Language Models. It features a refinement mechanism that automatically
attempts to subdivide categories if they contain too many items, promoting
more granular organization.

Key Techniques Showcased:
- Hierarchical categorization using LLMs.
- Iterative refinement of categories based on item count constraints.
- Use of Pydantic for defining structured LLM output.
- Employing LangChain's OutputFixingParser (or fallback) for robust parsing of LLM responses.
- Implementation of exponential backoff retries for handling API rate limits.

To Run:
1. Install required libraries:
   pip install langchain langchain-openai python-dotenv pydantic tenacity openai
2. Create a .env file in the same directory with your OPENAI_API_KEY:
   OPENAI_API_KEY=your_openai_api_key_here
3. Run the script: python standalone_hierarchical_categorizer.py
"""

import os
import json
import time
import copy
import re
import traceback
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from itertools import islice
from collections import defaultdict

# --- Third-party Libraries ---
import openai  # For RateLimitError specifically
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# --- LangChain Components ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# Ensure OutputFixingParser is available. If using newer langchain, import might change.
try:
    from langchain.output_parsers import OutputFixingParser
except ImportError:
    # Newer versions might use something different or require separate install
    # For now, provide a basic fallback or raise an error
    print("Warning: Could not import OutputFixingParser from langchain. Parsing might be less robust, especially for malformed JSON. Consider installing 'langchain>=0.1.17' or a compatible version if needed.")
    # Basic fallback: Use the base parser directly
    OutputFixingParser = PydanticOutputParser # Assign base parser as fallback
    def get_output_fixing_parser(parser, llm): # Wrapper function for compatibility
        """Fallback parser creator when OutputFixingParser is not available."""
        print("Using PydanticOutputParser as fallback for OutputFixingParser.")
        return parser
else:
    # If import succeeds, define a wrapper for consistency
    def get_output_fixing_parser(parser, llm):
        """Creates an OutputFixingParser instance."""
        print("Using OutputFixingParser for robust JSON handling.")
        return OutputFixingParser.from_llm(parser=parser, llm=llm)


from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

# ==============================================================================
# == CONFIGURATION
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not set. Please create a .env file.")

# --- Model & Processing Configuration ---
CATEGORIZATION_MODEL_NAME = "gpt-4o-mini" # Or "gpt-4", "gpt-3.5-turbo", etc.
LLM_TEMPERATURE = 0.1
MAX_ITEMS_PER_CHUNK_CAT = 50 # Max items processed in one LLM call for categorization/refinement
MAX_ITEMS_PER_CATEGORY_TARGET = 15 # Target max items per *final* category before refinement is triggered

MAX_REFINEMENT_ITERATIONS = 3 # Maximum number of refinement loops

# List of model prefixes known NOT to support the temperature parameter
MODELS_WITHOUT_TEMPERATURE = [
    "o3-mini", "o1-mini-2024-09-12", "o4-mini-2025-04-16"
]

# ==============================================================================
# == DATA MODELS (Generalized)
# ==============================================================================

class CategoryPath(BaseModel):
    model_config = {} # Add model_config for Pydantic v2 compatibility
    """
    Represents a hierarchical category path for an item.

    Attributes:
        path: A list of strings representing the category names from the root
              to the leaf category. An empty list indicates an uncategorized item.
    """
    path: List[str] = Field(
        default_factory=list,
        description="List of category names from root to leaf"
    )

    def __str__(self) -> str:
        """Returns the category path as a string with '>' separators."""
        return " > ".join(self.path)

    @property
    def main_category(self) -> str:
        """Returns the top-level category name or 'Uncategorized' if the path is empty."""
        return self.path[0] if self.path else "Uncategorized"

    @property
    def depth(self) -> int:
        """Returns the number of levels in the category path."""
        return len(self.path)

class DataItem(BaseModel):
    model_config = {} # Add model_config for Pydantic v2 compatibility
    """
    Represents a structured text item that needs to be categorized.

    Attributes:
        headline: A concise headline, summary, or identifier for the item.
        body: The main text content or supporting details of the item.
        citation: A reference to the source of the item (e.g., document name, URL, source + date).
        category_path: The hierarchical category path assigned to this item.
                       Defaults to an empty path (Uncategorized).
        url: Optional source URL, if available.
    """
    headline: str = Field(description="A concise headline, summary, or identifier for the item.")
    body: str = Field(description="The main text content or supporting details of the item.")
    citation: str = Field(description="A reference to the source of the item (e.g., document name, URL, source + date).")
    category_path: Optional[CategoryPath] = Field(
        default_factory=CategoryPath, # Ensures a default CategoryPath([]) is created
        description="Hierarchical category path assigned to this item."
    )
    url: Optional[str] = Field(default=None, description="Source URL, if available.")

    def get_unique_key(self) -> str:
        """
        Generates a unique key for the DataItem based on its headline and body content.
        This key is used for reliably mapping LLM outputs back to original items.

        Returns:
            A unique string key.
        """
        # Using headline and body hash for uniqueness
        body_hash = hashlib.md5(self.body.encode('utf-8')).hexdigest()
        return f"{self.headline}_{body_hash}"

class DataCollection(BaseModel):
    model_config = {} # Add model_config for Pydantic v2 compatibility
    """
    A collection of DataItems related to a specific subject or topic.

    Attributes:
        subject_name: The main subject or topic the data items relate to.
        items: A list of DataItem objects within this collection.
    """
    subject_name: str = Field(description="The main subject or topic the data items relate to.")
    items: List[DataItem]

    def get_category_tree(self) -> Dict[str, Dict]:
        """
        Builds a hierarchical tree representation of categories and their contained items.

        The tree structure is a nested dictionary where keys are category names
        and values are dictionaries containing 'items' (list of DataItem) and
        'subcategories' (another nested dictionary). Items with empty or no
        category paths are placed under "Uncategorized".

        Returns:
            A dictionary representing the hierarchical category tree.
        """
        tree = {}
        for item in self.items:
            current_level = tree
            # Ensure category_path exists and has a path list before accessing
            path_to_use = (item.category_path.path
                           if item.category_path and item.category_path.path
                           else ["Uncategorized"])

            # Iterate through the path, creating nodes if they don't exist
            for i, category_name in enumerate(path_to_use):
                # Ensure category name is treated as string key
                category_key = str(category_name) # Convert just in case
                if not category_key: # Skip empty category names
                    print(f"Warning: Skipping empty category name in path for item: {item.headline}")
                    continue

                if category_key not in current_level:
                    current_level[category_key] = {"items": [], "subcategories": {}}

                # If it's the last category in the path, add the item
                if i == len(path_to_use) - 1:
                    current_level[category_key]["items"].append(item.model_dump()) # Use model_dump() for JSON serialization
                else:
                    # Move deeper into the tree using the correct subcategory key
                    # Check if key exists before accessing subcategories
                    if category_key in current_level:
                        current_level = current_level[category_key]["subcategories"]
                    else:
                        # This case indicates an issue, maybe from empty name skip above
                        print(f"Warning: Could not find expected category key '{category_key}' during tree traversal for item: {item.headline}")
                        break # Stop traversing this path for this item
        return tree


# ==============================================================================
# == PROMPT TEMPLATES (Sanitized)
# ==============================================================================

# *** SANITIZED PROMPT ***
CATEGORIZATION_TEMPLATE = """
You are an AI assistant skilled in analyzing text and assigning hierarchical categories.
Your task is to categorize the provided text items about the subject: {subject_name}.

Assign a hierarchical category path to each item based on its content (headline). The path should go from a general topic to more specific ones (e.g., ["Topic A", "Subtopic A1", "Specific Detail A1a"] or ["Topic B", "Subtopic B1"]).

Guidelines:
1.  **Maximum Depth:** Limit paths to a maximum of 3 levels (e.g., ["Level 1", "Level 2", "Level 3"]).
2.  **Consistency:** Use consistent names for equivalent categories across different items (e.g., use "Performance Metrics" consistently).
3.  **Relevance:** The category path must accurately reflect the main subject of the item's headline/content.
4.  **Generic Categories (Examples - Adapt based on data):**
    *   Performance Metrics (-> Sales Data [-> Regional], User Engagement)
    *   Project Updates (-> Feature Development, Bug Fixes, Timeline Adjustments)
    *   Subject Attributes (-> Background Info, Key Statements, Related Events)
    *   External Factors (-> Market Trends, Competitor Actions, Regulatory Changes)
    *   Feedback (-> User Feedback, Stakeholder Input)
    *   Operational Issues (-> System Downtime, Resource Allocation)
    *   Uncategorized (Use ONLY if no other logical category fits well)
5.  **Granularity:** Be specific where possible within the 3-level limit. If an item is about a specific bug fix related to Feature Development, a path like ["Project Updates", "Feature Development", "Bug Fixes"] is preferred over just ["Project Updates", "Feature Development"].
6.  **Category Size Awareness:** If you notice a potential leaf category (e.g., ... "Specific Action Type") seems likely to contain more than {max_bullets_per_category} items based on the input chunk, consider if the Level 2 or Level 3 category could be slightly broader *while remaining accurate* for the specific item being categorized. Accuracy is the priority.

Provided Items (JSON list, containing headlines):
{items_json}

Output Instructions:
Return ONLY a valid JSON list of objects. Each object MUST contain:
- "headline": The exact headline string of the item provided in the input.
- "category_path": A JSON list of strings representing the hierarchical category path (e.g., ["Topic Group 1", "Subtopic 1A", "Detail 1A-i"]). Ensure the list is not empty unless truly uncategorizable.
{format_instructions}

Produce ONLY the raw JSON list starting with '[' and ending with ']'. Do not add any explanatory text before or after the list.
"""

# *** SANITIZED PROMPT ***
REFINEMENT_TEMPLATE = """
You are an AI assistant refining category assignments for better organization. A specific category path, '{category}', currently contains too many items ({num_bullets} items) and needs to be subdivided. The maximum target size per specific category is {max_bullets_per_category}.

Your task is to create **one** additional, more specific subcategory level for the items provided below, all of which currently share the parent path '{category}'.

IMPORTANT GUIDELINES:
1.  **Add One Level:** Create **only one** new level of subcategories nested under '{category}'.
2.  **Specificity:** The new subcategory names should clearly group similar items based on distinctions found in their headlines/content. Example: If '{category}' is ["Project Updates", "Feature Development"], new subcategories might be "UI Improvements", "Backend Logic", "API Changes".
3.  **Meaningful Grouping:** Aim for 2-5 logical subcategories that effectively distribute the items based on nuanced differences identified in the provided data. Avoid creating subcategories with only one item if possible, unless the distinction is very clear.
4.  **Accuracy:** Ensure the new, deeper path accurately reflects the content of each item. The new path MUST start with the original path segments from '{category}'.

Here are the items currently under the category '{category}' that need a more specific subcategory (JSON list, containing headlines):
{items_json}

Output Instructions:
Return ONLY a valid JSON list of objects. Each object MUST contain:
- "headline": The exact headline string of the original item.
- "category_path": A JSON list of strings representing the NEW, more specific, hierarchical category path. This path *must* be one level deeper than '{category}' and must start with the segments from '{category}'. Ensure the list is not empty.

Example Input Item (if category='["Metrics", "User Engagement"]'):
{{ "headline": "User sign-ups increased by 10%." }}

Example Output Object:
{{ "headline": "User sign-ups increased by 10%.", "category_path": ["Metrics", "User Engagement", "Acquisition"] }}

{format_instructions}

Produce ONLY the raw JSON list starting with '[' and ending with ']'. Do not add any explanatory text before or after the list.
"""


# ==============================================================================
# == Pydantic Models for Parsing LLM Output
# ==============================================================================

class CategorizationItem(BaseModel):
    model_config = {} # Add model_config for Pydantic v2 compatibility
    """Structure for a single categorized item output by the LLM."""
    headline: str = Field(description="The exact headline/identifier of the item.")
    category_path: List[str] = Field(description="The hierarchical category path assigned.")

class CategorizationList(BaseModel):
    model_config = {} # Add model_config for Pydantic v2 compatibility
    """Structure for the list of categorized items output by the LLM."""
    items: List[CategorizationItem] = Field(description="A list of categorized items.")

# ==============================================================================
# == HELPER FUNCTIONS
# ==============================================================================

def _log_retry_attempt(retry_state):
    """
    Logs details of a retry attempt when a RateLimitError occurs during an LLM call.

    Args:
        retry_state: The state object provided by Tenacity.
    """
    exc = retry_state.outcome.exception()
    wait_time = getattr(retry_state.next_action, 'sleep', 0)
    print(f"RateLimitError encountered: {exc}. Retrying attempt {retry_state.attempt_number} after {wait_time:.2f} seconds...")

def _sanitize_category_name(name: Any) -> str:
    """
    Sanitizes a category name by stripping whitespace and applying title case.
    Handles non-string inputs gracefully.

    Args:
        name: The raw category name to sanitize.

    Returns:
        A sanitized string suitable for use as a category name. Returns an empty
        string if the input is effectively empty after stripping.
    """
    # Convert to string first to handle potential non-string inputs from LLM
    str_name = str(name).strip()
    if not str_name:
        return "" # Return empty string if input is empty or just whitespace
    # Apply title case for consistency
    return str_name.title()

# ==============================================================================
# == MAIN CATEGORIZER CLASS (Generalized)
# ==============================================================================

class HierarchicalCategorizer:
    """
    Assigns hierarchical categories to structured text items using an LLM and
    iteratively refines categories that exceed a defined size threshold.

    This class orchestrates the categorization process, including chunking items,
    calling the LLM for initial categorization, parsing the results, and
    performing iterative refinement of oversized categories.

    Attributes:
        model_name (str): Name of the OpenAI model.
        temperature (float): LLM temperature setting.
        llm (ChatOpenAI): Initialized LangChain LLM instance.
        base_parser (PydanticOutputParser): Base parser for structured output.
        fixing_parser (Runnable): Robust parser (potentially OutputFixingParser).
        categorization_prompt (PromptTemplate): Prompt for initial categorization.
        refinement_prompt (PromptTemplate): Prompt for category refinement.
        categorization_chain (Runnable): Chain for initial categorization.
        refinement_chain (Runnable): Chain for category refinement.
    """

    def __init__(self,
                 model_name: str = CATEGORIZATION_MODEL_NAME,
                 temperature: float = LLM_TEMPERATURE):
        """
        Initializes the HierarchicalCategorizer.

        Args:
            model_name (str): The name of the OpenAI model to use.
            temperature (float): The temperature for the LLM.

        Raises:
            ValueError: If the OPENAI_API_KEY is not set.
        """
        self.model_name = model_name
        self.temperature = temperature

        # --- Set up the LLM ---
        llm_kwargs = {"model_name": self.model_name, "openai_api_key": OPENAI_API_KEY}
        supports_temperature = not any(
            self.model_name.startswith(prefix) for prefix in MODELS_WITHOUT_TEMPERATURE
        )
        if supports_temperature:
            llm_kwargs["temperature"] = self.temperature
            print(f"Initializing LLM: {self.model_name} with temperature={self.temperature}")
        else:
             print(f"Initializing LLM: {self.model_name} without temperature (not supported).")
        try:
            self.llm = ChatOpenAI(**llm_kwargs)
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {e}")
            raise

        # --- Parser Setup ---
        self.base_parser = PydanticOutputParser(pydantic_object=CategorizationList)
        # Use the wrapper function to handle potential import differences
        self.fixing_parser = get_output_fixing_parser(self.base_parser, self.llm)

        # --- Create Prompt Templates ---
        self.categorization_prompt = PromptTemplate(
            template=CATEGORIZATION_TEMPLATE,
            input_variables=["subject_name", "items_json"],
            partial_variables={
                "format_instructions": self.base_parser.get_format_instructions(),
                "max_bullets_per_category": MAX_ITEMS_PER_CATEGORY_TARGET
            }
        )
        self.refinement_prompt = PromptTemplate(
            template=REFINEMENT_TEMPLATE,
            input_variables=["category", "num_bullets", "items_json"],
            partial_variables={
                "format_instructions": self.base_parser.get_format_instructions(),
                "max_bullets_per_category": MAX_ITEMS_PER_CATEGORY_TARGET
            }
        )

        # --- Create Chains ---
        self.categorization_chain: Runnable = self.categorization_prompt | self.llm | self.fixing_parser
        self.refinement_chain: Runnable = self.refinement_prompt | self.llm | self.fixing_parser

        print("--- HierarchicalCategorizer Initialization Complete ---")
        print(f"  Model: {self.model_name}")
        print(f"  Max Items per LLM Chunk: {MAX_ITEMS_PER_CHUNK_CAT}")
        print(f"  Refinement Trigger Threshold: > {MAX_ITEMS_PER_CATEGORY_TARGET} items per category path")
        print("---------------------------------------------------------")

    def _chunk_items(self, items: List[DataItem], chunk_size: int) -> List[List[DataItem]]:
        """
        Splits a list of DataItem objects into smaller chunks.

        Args:
            items: The list of DataItem objects to chunk.
            chunk_size: The maximum number of items per chunk.

        Returns:
            A list of lists, where each inner list is a chunk.
        """
        if not items: return []
        chunks = []
        iterator = iter(items)
        while chunk := list(islice(iterator, chunk_size)):
            chunks.append(chunk)
        return chunks

    @retry(
        wait=wait_random_exponential(min=5, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=_log_retry_attempt
    )
    def _invoke_with_retry(self, chain: Runnable, input_dict: Dict) -> Any:
        """
        Invokes a Langchain chain with retries on RateLimitError.

        Args:
            chain: The Langchain Runnable chain to invoke.
            input_dict: The input dictionary for the chain.

        Returns:
            The result from the chain invocation.

        Raises:
            openai.RateLimitError: If retries are exhausted.
            Exception: Other exceptions from the chain.
        """
        return chain.invoke(input_dict)

    def _categorize_item_chunk_inplace(self,
                                         item_chunk: List[DataItem],
                                         subject_name: str,
                                         item_map: Dict[str, DataItem]):
        """
        Categorizes a chunk of items and updates the main item map in-place.

        Args:
            item_chunk: The list of DataItem objects for this chunk.
            subject_name: The subject name for the prompt context.
            item_map: The main dictionary mapping unique keys to DataItem objects.
        """
        # Only include headlines for the categorization prompt
        items_for_prompt = [{"headline": item.headline} for item in item_chunk if item.headline] # Ensure headline exists
        if not items_for_prompt:
            print("  Skipping empty or headline-less categorization chunk.")
            return

        items_json = json.dumps(items_for_prompt, ensure_ascii=False, indent=2)

        try:
            print(f"  Invoking categorization chain for chunk ({len(items_for_prompt)} items)...")
            invoke_input = {"subject_name": subject_name, "items_json": items_json}
            # The chain includes the fixing parser
            result = self._invoke_with_retry(self.categorization_chain, invoke_input)

            print(f"  Categorization chain invoked successfully.")
            if not isinstance(result, CategorizationList):
                print(f"  Warning: Categorization chain returned unexpected type: {type(result)}. No updates for this chunk.")
                return # Exit if result is not as expected

            categorization_items = result.items
            updated_count = 0
            skipped_count = 0
            # Map headline to unique key for efficient lookup within this chunk
            headline_map_chunk = {item.headline: item.get_unique_key() for item in item_chunk}

            for cat_data in categorization_items:
                # Validate the structure of the received item
                if not isinstance(cat_data, CategorizationItem):
                    print(f"  Warning: Skipping item in received list due to incorrect type: {type(cat_data).__name__}. Data: {cat_data}")
                    skipped_count += 1
                    continue

                headline = cat_data.headline
                # Handle both path and category_path fields from LLM response
                category_path_list = getattr(cat_data, 'path', []) or cat_data.category_path

                # Basic validation of parsed data content
                if not headline or not category_path_list or not isinstance(category_path_list, list) or not category_path_list:
                    print(f"  Warning: Skipping categorization due to missing/invalid data from LLM. Headline: '{headline}', Path: '{category_path_list}'. Path must be a non-empty list.")
                    skipped_count += 1
                    continue

                # Find the original item's unique key using the headline
                item_key = headline_map_chunk.get(headline)
                if item_key and item_key in item_map:
                    # Sanitize and Update the category path in the main map
                    sanitized_path = [_sanitize_category_name(name) for name in category_path_list if _sanitize_category_name(name)] # Filter out empty names after sanitizing
                    if not sanitized_path: # If sanitization resulted in an empty path
                         print(f"  Warning: Skipping update for headline '{headline}' because sanitized path is empty.")
                         skipped_count += 1
                         continue
                    item_map[item_key].category_path = CategoryPath(path=sanitized_path)
                    updated_count += 1
                else:
                    # Log if the LLM returned a headline not present in the input chunk
                    print(f"  Warning: Could not find/map original item for headline returned by LLM: '{headline}'. Item not found in the original chunk.")
                    skipped_count += 1

            print(f"  Chunk categorization update complete: {updated_count} updated, {skipped_count} skipped/not found.")

        except OutputParserException as e:
            print(f"  Error: Parser failed for categorization chunk: {str(e)[:300]}...")
        except openai.RateLimitError:
             print("  Error: Rate limit exceeded during categorization chunk processing after retries.")
             # Decide how to handle - skip chunk? raise error?
        except Exception as e:
            print(f"  Error during categorization chunk processing: {type(e).__name__} - {e}")
            traceback.print_exc()

    def _refine_category_llm_call(self,
                                  category_path_str: str,
                                  items: List[DataItem]) -> Optional[List[CategorizationItem]]:
        """
        Performs LLM call to refine a category.

        Args:
            category_path_str: String representation of the category path to refine.
            items: List of DataItems in this category needing subdivision.

        Returns:
            Optional list of CategorizationItem objects with refined paths, or None on failure.
        """
        # Only include headlines for the refinement prompt
        items_for_prompt = [{"headline": item.headline} for item in items if item.headline]
        if not items_for_prompt:
            print(f"  Skipping refinement call for '{category_path_str}': No items with headlines found.")
            return [] # Return empty list, not None

        items_json = json.dumps(items_for_prompt, ensure_ascii=False, indent=2)

        print(f"  Invoking refinement chain for category '{category_path_str}' ({len(items)} items)...")
        try:
            invoke_input = {
                "category": category_path_str,
                "num_bullets": len(items), # Keep param name as num_bullets for prompt compatibility
                "items_json": items_json
            }
            # The chain includes the fixing parser
            result = self._invoke_with_retry(self.refinement_chain, invoke_input)
            print(f"  Refinement chain invoked successfully for '{category_path_str}'.")

            if isinstance(result, CategorizationList):
                 return result.items # Return the list of CategorizationItem objects
            else:
                 print(f"  Warning: Refinement chain did not return CategorizationList for '{category_path_str}'. Got: {type(result)}")
                 return None

        except OutputParserException as e:
            print(f"  Error: Parser failed for refinement of '{category_path_str}': {str(e)[:300]}...")
            return None
        except openai.RateLimitError:
            print(f"  Error: Rate limit exceeded during refinement call for '{category_path_str}' after retries.")
            return None
        except Exception as e:
            print(f"  Error during refinement LLM call for '{category_path_str}': {type(e).__name__} - {e}")
            traceback.print_exc()
            return None

    def _refine_category_distribution(self, data_collection: DataCollection) -> DataCollection:
        """
        Iteratively refines categories exceeding the target item count.

        Args:
            data_collection: The DataCollection object containing items to refine.

        Returns:
            A new DataCollection object with potentially refined category paths.
        """
        iteration = 0
        max_iterations = MAX_REFINEMENT_ITERATIONS
        # Work on a mutable map of items, ensuring deep copies initially
        item_map = {item.get_unique_key(): copy.deepcopy(item) for item in data_collection.items}

        while iteration < max_iterations:
            iteration += 1
            print(f"--- Refinement Iteration {iteration}/{max_iterations} ---")
            needs_refinement_this_iter = False
            category_counts = defaultdict(int)
            category_items_map = defaultdict(list)

            # Recalculate counts based on current item categories in the map
            current_items = list(item_map.values())
            for item in current_items:
                # Use tuple for dictionary key, handle uncategorized/empty path
                path_tuple = tuple(item.category_path.path) if (item.category_path and item.category_path.path) else ("Uncategorized",)
                category_counts[path_tuple] += 1
                category_items_map[path_tuple].append(item)

            # Identify categories needing refinement in this iteration
            categories_to_refine = []
            for category_path_tuple, count in category_counts.items():
                # Refine if count exceeds TARGET and path depth allows subdivision (< 3)
                # Also skip refining the ("Uncategorized",) path
                if count > MAX_ITEMS_PER_CATEGORY_TARGET and len(category_path_tuple) < 3 and category_path_tuple != ("Uncategorized",):
                    categories_to_refine.append((category_path_tuple, count))

            if not categories_to_refine:
                print("No categories require further refinement in this iteration.")
                break # Finished refining

            needs_refinement_this_iter = True
            categories_to_refine.sort(key=lambda x: x[1], reverse=True) # Process largest first

            print(f"Found {len(categories_to_refine)} categories needing refinement (> {MAX_ITEMS_PER_CATEGORY_TARGET} items):")
            for cat_tuple, count in categories_to_refine:
                 print(f"  - '{' > '.join(cat_tuple)}' ({count} items)")

            # Refine each identified category
            for category_path_tuple, count in categories_to_refine:
                category_path_str = " > ".join(category_path_tuple)
                print(f"\nRefining category: '{category_path_str}' ({count} items)...")
                items_to_refine = category_items_map[category_path_tuple]

                # Call LLM for refinement suggestions
                refined_categorizations = self._refine_category_llm_call(
                    category_path_str, items_to_refine
                )

                # Apply refinements back to the main item map
                update_count = 0
                skipped_count = 0
                # Create a quick lookup for headlines within the refinement set
                headline_map_refine = {item.headline: item.get_unique_key() for item in items_to_refine}

                if refined_categorizations is None: # Check for None explicitly
                    print(f"  Refinement LLM call for '{category_path_str}' failed or returned no valid data. Skipping updates for this category.")
                    continue # Skip to next category to refine

                for ref_item in refined_categorizations: # ref_item is CategorizationItem
                    # Validate the structure of the received item
                    if not isinstance(ref_item, CategorizationItem):
                        print(f"  Warning: Skipping item in refinement list due to incorrect type: {type(ref_item).__name__}. Data: {ref_item}")
                        skipped_count += 1
                        continue

                    headline = ref_item.headline
                    new_path_list = ref_item.category_path

                    # Basic validation of parsed data content
                    if not headline or not new_path_list or not isinstance(new_path_list, list) or not new_path_list:
                        print(f"  Warning: Skipping refinement update due to missing/invalid data from LLM. Headline: '{headline}', Path: '{new_path_list}'. Path must be a non-empty list.")
                        skipped_count += 1
                        continue

                    # Find the original item's unique key using the headline
                    item_key = headline_map_refine.get(headline)
                    if item_key and item_key in item_map:
                        original_path_len = len(item_map[item_key].category_path.path) if item_map[item_key].category_path else 0
                        # Sanitize the new path from LLM
                        sanitized_new_path = [_sanitize_category_name(name) for name in new_path_list if _sanitize_category_name(name)]

                        if not sanitized_new_path:
                            print(f"  Warning: Skipping refinement update for item '{headline[:50]}...': Sanitized new path is empty.")
                            skipped_count += 1
                            continue

                        # Check if the new path is valid (starts with original) and is strictly deeper
                        if (len(sanitized_new_path) > original_path_len and
                            list(category_path_tuple) == sanitized_new_path[:len(category_path_tuple)]):
                            # Directly update the path attribute of the existing CategoryPath object
                            item_map[item_key].category_path.path = sanitized_new_path
                            update_count += 1
                        else:
                            # Log why refinement was skipped
                            print(f"  - Refinement skipped for item '{headline[:50]}...': New path '{sanitized_new_path}' not valid or not deeper than original '{list(category_path_tuple)}'")
                            skipped_count += 1
                    else:
                        # Log if LLM returned a headline not in the refinement set
                        print(f"  Warning: Could not find/map original item for headline returned by LLM during refinement: '{headline}'.")
                        skipped_count += 1

                print(f"  Refinement update complete for '{category_path_str}': {update_count} items updated, {skipped_count} skipped/unchanged.")
            # -- End loop refining categories for this iteration --

        if iteration >= max_iterations and needs_refinement_this_iter:
            print(f"Warning: Reached maximum refinement iterations ({max_iterations}). Some categories might still exceed target size {MAX_ITEMS_PER_CATEGORY_TARGET}.")

        # Return a new collection based on the final state of the item map
        return DataCollection(
            subject_name=data_collection.subject_name,
            items=list(item_map.values())
        )

    def categorize_data_collection(self, data_collection: DataCollection) -> DataCollection:
        """
        Assigns hierarchical categories to items in a DataCollection, including refinement.
        This is the main public method to categorize a collection of items.

        Args:
            data_collection: The DataCollection object containing items to categorize.

        Returns:
            A new DataCollection object with items assigned to hierarchical categories,
            potentially refined based on item distribution.
        """
        total_items = len(data_collection.items)
        print(f"\nStarting categorization process for {total_items} items for subject '{data_collection.subject_name}'...")

        # Create the initial map using deep copies to avoid side effects
        initial_item_map = {item.get_unique_key(): copy.deepcopy(item) for item in data_collection.items}

        # --- Initial Categorization Phase ---
        print("\n--- Starting Initial Categorization Phase ---")
        # Chunking logic for initial categorization
        if total_items <= MAX_ITEMS_PER_CHUNK_CAT:
            print(f"Processing all {total_items} items in a single categorization chunk.")
            # Pass the list of items from the map for processing
            self._categorize_item_chunk_inplace(list(initial_item_map.values()), data_collection.subject_name, initial_item_map)
        else:
            # Chunk based on the original list's order for consistency
            # Need to map chunks back to keys or pass map slice? Let's chunk keys.
            all_keys = list(initial_item_map.keys())
            key_chunks = [all_keys[i:i + MAX_ITEMS_PER_CHUNK_CAT] for i in range(0, total_items, MAX_ITEMS_PER_CHUNK_CAT)]

            print(f"Processing {len(key_chunks)} categorization chunks of up to {MAX_ITEMS_PER_CHUNK_CAT} items each...")
            for i, key_chunk in enumerate(key_chunks):
                print(f"-- Processing categorization chunk {i+1}/{len(key_chunks)} --")
                # Get the actual items for this chunk
                item_chunk = [initial_item_map[key] for key in key_chunk if key in initial_item_map]
                # Pass the item chunk to categorize, updates happen in the main map
                self._categorize_item_chunk_inplace(item_chunk, data_collection.subject_name, initial_item_map)
        print("--- Initial Categorization Phase Complete ---")

        # Create an intermediate collection using the updated map values
        intermediate_collection = DataCollection(
            subject_name=data_collection.subject_name,
            items=list(initial_item_map.values())
        )

        # --- Refinement Stage ---
        print("\n--- Starting Refinement Stage ---")
        # Pass the collection with initial categories to the refinement process
        refined_collection = self._refine_category_distribution(intermediate_collection)
        print("--- Refinement Stage Complete ---")

        category_tree = refined_collection.get_category_tree()
        print(f"\nCategorization & Refinement completed. Final stats:")
        print(f"  - Total Items Processed: {len(refined_collection.items)}")
        print(f"  - Top-Level Categories Found: {len(category_tree)}")
        return refined_collection


# ==============================================================================
# == EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Running standalone hierarchical categorizer example...")

    if not OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please create a .env file with OPENAI_API_KEY='your-key' to run the example.")
    else:
        # --- Create Sample Generic Data ---
        # Data designed to potentially trigger refinement in "Metrics"
        sample_items_base = [
            DataItem(headline="Metric A Increased Significantly.", body="Details about Metric A's increase...", citation="[Report X, Q1 2024]"),
            DataItem(headline="Metric B Showed Slight Decline.", body="Analysis of Metric B's trend...", citation="[Dashboard Y, 1/15/24]"),
            DataItem(headline="Feature X Development Completed.", body="Feature X passed QA testing...", citation="[Jira Ticket-123, 1/10/24]"),
            DataItem(headline="Bug Found In Feature Y.", body="Critical bug reported affecting users...", citation="[Support Log Z, 1/11/24]"),
            DataItem(headline="Timeline Adjusted For Project Alpha.", body="Project Alpha deadline moved due to dependencies...", citation="[Meeting Notes, 1/12/24]"),
            DataItem(headline="New Competitor Product Launched.", body="Competitor C launched a competing product...", citation="[Market News, 1/5/24]"),
            DataItem(headline="User Feedback Received On Feature X.", body="Positive feedback received via survey...", citation="[Survey Results, 1/14/24]"),
            DataItem(headline="Metric A Correlated With Marketing Campaign.", body="Analysis shows correlation with Campaign Z...", citation="[Analysis Report, Q1 2024]"),
            DataItem(headline="Team Meeting Discussed Metric B.", body="Action items created to address decline...", citation="[Meeting Notes, 1/16/24]"),
            DataItem(headline="Regression Bug Fixed In Feature Y.", body="Patch deployed to production...", citation="[Jira Ticket-125, 1/18/24]"),
            DataItem(headline="Project Alpha Phase 2 Started.", body="Phase 2 kickoff meeting held...", citation="[Project Plan, 1/19/24]"),
            DataItem(headline="Expert Praised Metric A Results.", body="Quote from expert on the significance...", citation="[Internal Comms, 1/20/24]"),
            DataItem(headline="Feature X User Adoption Rate Tracked.", body="Adoption rate at 20% after first week...", citation="[Analytics Dashboard, 1/17/24]"),
            DataItem(headline="Dependency Issue Resolved For Project Alpha.", body="External team delivered required component...", citation="[Email Update, 1/21/24]"),
            DataItem(headline="Market Share Analysis Completed.", body="Our market share remained stable...", citation="[Market Report Q4 2023, 1/22/24]"),
            DataItem(headline="A/B Test Results For Feature Z Ready.", body="Variant B showed higher conversion...", citation="[AB Test Platform, 1/23/24]"),
            DataItem(headline="Support Tickets Related To Bug Decreased.", body="Ticket volume dropped after patch...", citation="[Support System, 1/24/24]"),
            # Add more items likely to fall under "Metrics" to test refinement
            DataItem(headline="Metric C (User Retention) Improved.", body="Retention rate up by 5%...", citation="[Dashboard Y, 1/25/24]"),
            DataItem(headline="Daily Active Users (Metric D) Stable.", body="DAU shows consistent pattern...", citation="[Analytics, 1/26/24]"),
            DataItem(headline="Conversion Rate (Metric E) Met Target.", body="Q1 conversion rate target achieved...", citation="[Sales Report, Q1 2024]"),
            DataItem(headline="Churn Rate (Metric F) Needs Attention.", body="Churn rate slightly increased month-over-month...", citation="[Dashboard Y, 1/27/24]"),
            DataItem(headline="Average Session Duration (Metric G) Up.", body="Users spending more time per session...", citation="[Analytics, 1/28/24]"),
        ]
        sample_items = sample_items_base # Use the base list (adjust multiplier if needed for testing)
        print(f"Created {len(sample_items)} sample data items.")

        sample_data_collection = DataCollection(subject_name="Project Performance Metrics and Updates", items=sample_items)

        # --- Initialize and Run Categorizer ---
        print("\nInitializing categorizer...")
        categorizer = HierarchicalCategorizer()
        start_time = time.time()
        # Use the main public method
        categorized_collection = categorizer.categorize_data_collection(sample_data_collection)
        end_time = time.time()
        print(f"\nTotal categorization time: {end_time - start_time:.2f} seconds")

        # --- Print Results ---
        print("\n--- Categorization Results ---")
        print(f"Subject: {categorized_collection.subject_name}")
        for i, item in enumerate(categorized_collection.items):
            # Ensure category_path exists before string conversion
            path_str = str(item.category_path) if item.category_path else "Uncategorized"
            print(f"{i+1}. Headline: {item.headline}")
            print(f"   Category: {path_str}")
            print("-" * 15)

        # Optionally print the category tree structure for verification
        print("\n--- Final Category Tree Structure ---")
        tree = categorized_collection.get_category_tree()
        import pprint
        # Use sort_dicts=False if using Python 3.8+ and wanting insertion order preserved visually
        pprint.pprint(tree, indent=2, width=120, sort_dicts=False)
if __name__ == "__main__":
    print("\n--- Running Demonstration ---")

    # Create some sample data items
    sample_items = [
        DataItem(headline="Meeting notes on Q3 performance", body="Discussed sales figures and user engagement metrics.", citation="Meeting 2024-10-26"),
        DataItem(headline="Bug report: Login failed on mobile", body="Users are unable to log in using the Android app.", citation="Bugzilla #12345"),
        DataItem(headline="Feature request: Dark mode option", body="Users have requested a dark mode for the web application.", citation="User Feedback Form"),
        DataItem(headline="Competitor analysis: New pricing strategy", body="Analyzing the recent pricing changes by a major competitor.", citation="Market Research Report"),
        DataItem(headline="Meeting notes on Q3 performance follow-up", body="Decided on action items based on the Q3 performance discussion.", citation="Meeting 2024-10-28"),
        DataItem(headline="Server downtime incident", body="Unexpected downtime occurred on the production server.", citation="Incident Report #987"),
    ]

    # Create a DataCollection
    collection_to_categorize = DataCollection(subject_name="Project Overview", items=sample_items)

    # Instantiate the categorizer
    # Use a dummy model name as we are just demonstrating the flow
    categorizer = HierarchicalCategorizer(model_name="gpt-4o-mini")

    # Categorize the data collection
    categorized_collection = categorizer.categorize_data_collection(collection_to_categorize)

    print("\n--- Categorization Complete ---")
    print("\nCategory Tree:")
    # Print the category tree
    import json
    print(json.dumps(categorized_collection.get_category_tree(), indent=2))

    print("\n--- Demonstration Complete ---")