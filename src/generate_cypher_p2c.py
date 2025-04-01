from openai import OpenAI
import re
import os
from dotenv import load_dotenv
load_dotenv()

def format_as_list(response):
    """
    Formats the response as a list.

    Args:
        response (str): The response to format.
    Returns:
        list: The formatted response as a list.
    """
    clean_response = re.sub(r'[^a-zA-Z0-9,]', '', response)  # Remove all characters except letters, numbers, and commas
    return clean_response.split(",")

def read_file_content(file_path):
    """
    Reads content from a file.

    Args:
        file_path (str): Path to the file.
    Returns:
        str: Content of the file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")

def generate_cypher_query(user_query, llm_config, schema, instructions_path, max_attempts=5) -> str:
    """
    Generates a Cypher query based on the user's query, with syntax and semantic validations.

    Args:
        user_query (str): The user's query.
        llm_config: The LLM configuration.
        schema: The schema to use for generating the Cypher query.
        instructions_path: The path to the instructions file.
        max_attempts (int, optional): The maximum number of attempts to generate the Cypher query. Defaults to 5.

    Returns:
        str: The generated Cypher query.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    list_of_nodes = [] 
    for node in list(schema["labels"].keys()):
        if node in schema["nodes_description"]:
            node = f"'{node}' ({schema['nodes_description'][node]})"
        list_of_nodes.append(node)
    try:
        # Read node relevance instructions from file
        node_relevance_instructions = read_file_content('./prompts/node_relevance_instructions.txt')
        node_relevance_instructions = node_relevance_instructions.format(
            user_query=user_query,
            list_of_nodes=list_of_nodes
        )
        
        # Step 1: Identify relevant nodes
        nodes_prompt = f"Instructions: {node_relevance_instructions}"
        messages = [{"role": "user", "content": nodes_prompt}]
        response = client.chat.completions.create(model=llm_config["config_list"][0]["litemodel"], messages=messages)
        relevant_nodes = format_as_list(response.choices[0].message.content.strip())
        
        # Step 2: Identify relevant relationships
        list_of_relationships = schema.get("relationships", [])
  
        relevant_relationships = [
            r for r in list_of_relationships
            if r["from"] in relevant_nodes or r["to"] in relevant_nodes
        ]
        
        # Add the related comment to each relationship
        for r in relevant_relationships:
            r['comment'] = schema['relationship_comments'].get(r['name'], '')
        
        # Step 3: Add nodes description and properties descriptions to the relevant nodes
        relevant_nodes_info = {}
        for node in relevant_nodes:
            node_description = schema['nodes_description'].get(node, "")
            node_properties = schema['nodes_properties'].get(node, {})
            relevant_nodes_info[node] = {
                'description': node_description,
                'properties': node_properties,
            }

        # Prepare nodes information as a formatted string
        nodes_info_str = ""
        for node, info in relevant_nodes_info.items():
            nodes_info_str += f"Node: {node}\nDescription: {info['description']}\nProperties:\n"
            for prop, prop_desc in info['properties'].items():
                nodes_info_str += f"- {prop}: {prop_desc}\n"
            nodes_info_str += "\n"

        # Step 4: Generate the Cypher query
        with open(instructions_path, "r") as file:
            instructions = file.read()

        final_prompt = (
            f"You are a helpful AI assistant that generates Cypher queries to answer user queries.\n"
            f"Generate a Cypher query to answer the User query: ```{user_query}```\n"
            f"Consider the following information:\n"
            f"Given the relevant nodes ({relevant_nodes}) and all the relationships they are involved in: ```{relevant_relationships}```\n"
            f"Information about the relevant nodes:\n{nodes_info_str}\n"
            f"\nIMPORTANT Instructions: {instructions}\n"
            f"The Cypher query should answer the user query which is: {user_query}\n"
            f"Return only Cypher code with no explanations. Complete this task: the cypher query is: ...")
                
        response = client.chat.completions.create(
            model=llm_config["config_list"][0]["model"],
            messages=[{"role": "user", "content": final_prompt}]
        )
        cypher_query = response.choices[0].message.content.strip().replace("\n", " ")

    except Exception as e:
        cypher_query = "Error in generating Cypher query: " + str(e)

    return cypher_query