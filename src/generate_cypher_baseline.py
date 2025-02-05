"""
This script contains the functions to generate a Cypher query based on the user's query. (Baseline Approach)
This version does not use relevant node information and calls the LLM only once.
"""
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def reduce_context_size(context):
    """
    Reduces the context size by removing extra characters.

    Args:
        context (str): The context to reduce.
    Returns:
        str: The reduced context.
    """
    return context

def generate_cypher_query(user_query, llm_config, schema, instructions_path, max_attempts=5) -> str:
    """
    Generates a Cypher query based on the user's query, without using relevant node information.

    Args:
        user_query (str): The user's query.
        llm_config: The LLM configuration.
        schema: The schema to use for generating the Cypher query.
        instructions_path: The path to the instructions file.

    Returns:
        str: The generated Cypher query.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # Read the instructions from the specified file
        # with open(instructions_path.replace("[VERSION]", "v6"), "r") as file:
        #     instructions = file.read()
        instructions = "- Limit the results to 100 entries (append `LIMIT 100` to the query)." \
                        "\n - Use `DISTINCT` to avoid duplicate results whenever appropriate." \
                        "\n - Return only the Cypher query without any explanations or comments."

        # Construct the final prompt as per your specification
        final_prompt = (
            f"You are a helpful AI assistant that generates Cypher queries to answer user queries.\n"
            f"Generate a Cypher query to answer the User query: ```{user_query}```\n"
            f"The schema of the Knowledge Graph is as follows:\nLabels:{schema['labels']}\nRelationships:{schema['relationships']}\n"
            f"\nIMPORTANT Instructions: {instructions}\n"
            f"The Cypher query should answer the user query which is: {user_query}\n"
            f"Return only Cypher code with no explanations. Complete this task: the cypher query is: ..."
        )

        # Send the prompt to the LLM
        messages = [{"role": "user", "content": reduce_context_size(final_prompt)}]
        response = client.chat.completions.create(
            model=llm_config["config_list"][0]["model"],
            messages=messages
        )

        # Extract the Cypher query from the LLM response
        cypher_query = response.choices[0].message.content.strip().replace("\n", " ")

    except Exception as e:
        cypher_query = "Error in generating Cypher query: " + str(e)

    return cypher_query
