from openai import OpenAI
import re
import os
import json
import time
import datetime
from typing import Dict, List, Any, Tuple
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

def generate_with_instructions(user_query: str, llm_config: Dict, schema: Dict, instructions_path: str) -> str:
    """
    Generates a Cypher query with instructions.

    Args:
        user_query (str): The user's query.
        llm_config (Dict): The LLM configuration.
        schema (Dict): The schema information.
        instructions_path (str): Path to the instructions file.

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

        # Step 4: Generate the Cypher query with instructions
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

def generate_without_instructions(user_query: str, llm_config: Dict, schema: Dict, _: str) -> str:
    """
    Generates a Cypher query without instructions.

    Args:
        user_query (str): The user's query.
        llm_config (Dict): The LLM configuration.
        schema (Dict): The schema information.
        _ (str): Placeholder for instructions_path (not used).

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

        # Step 4: Generate the Cypher query without instructions
        final_prompt = (
            f"You are a helpful AI assistant that generates Cypher queries to answer user queries.\n"
            f"Generate a Cypher query to answer the User query: ```{user_query}```\n"
            f"Consider the following information:\n"
            f"Given the relevant nodes ({relevant_nodes}) and all the relationships they are involved in: ```{relevant_relationships}```\n"
            f"Information about the relevant nodes:\n{nodes_info_str}\n"
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

def generate_without_schema_comments(user_query: str, llm_config: Dict, schema: Dict, instructions_path: str) -> str:
    """
    Generates a Cypher query without schema comments.

    Args:
        user_query (str): The user's query.
        llm_config (Dict): The LLM configuration.
        schema (Dict): The schema information.
        instructions_path (str): Path to the instructions file.

    Returns:
        str: The generated Cypher query.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a schema copy without comments
    schema_no_comments = schema.copy()
    schema_no_comments["nodes_description"] = {}
    schema_no_comments["nodes_properties"] = {}
    schema_no_comments["relationship_comments"] = {}
    
    list_of_nodes = list(schema_no_comments["labels"].keys())
    
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
        list_of_relationships = schema_no_comments.get("relationships", [])
        relevant_relationships = [
            r for r in list_of_relationships
            if r["from"] in relevant_nodes or r["to"] in relevant_nodes
        ]
        
        # Step 3: Generate the Cypher query
        with open(instructions_path, "r") as file:
            instructions = file.read()

        final_prompt = (
            f"You are a helpful AI assistant that generates Cypher queries to answer user queries.\n"
            f"Generate a Cypher query to answer the User query: ```{user_query}```\n"
            f"Consider the following information:\n"
            f"Given the relevant nodes ({relevant_nodes}) and all the relationships they are involved in: ```{relevant_relationships}```\n"
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

def generate_without_relevant_nodes(user_query: str, llm_config: Dict, schema: Dict, instructions_path: str) -> str:
    """
    Generates a Cypher query without filtering for relevant nodes.

    Args:
        user_query (str): The user's query.
        llm_config (Dict): The LLM configuration.
        schema (Dict): The schema information.
        instructions_path (str): Path to the instructions file.

    Returns:
        str: The generated Cypher query.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Use all nodes instead of only relevant ones
    all_nodes = list(schema["labels"].keys())
    
    try:
        # Step 1: Use all nodes
        relevant_nodes = all_nodes
        
        # Step 2: Identify all relationships
        list_of_relationships = schema.get("relationships", [])
        
        # Add the related comment to each relationship
        for r in list_of_relationships:
            r['comment'] = schema['relationship_comments'].get(r['name'], '')
        
        # Step 3: Add nodes description and properties descriptions to all nodes
        nodes_info = {}
        for node in all_nodes:
            node_description = schema['nodes_description'].get(node, "")
            node_properties = schema['nodes_properties'].get(node, {})
            nodes_info[node] = {
                'description': node_description,
                'properties': node_properties,
            }

        # Prepare nodes information as a formatted string
        nodes_info_str = ""
        for node, info in nodes_info.items():
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
            f"Consider the following information about the knowledge graph:\n"
            f"Here are all the nodes and relationships in the knowledge graph.\n"
            f"Nodes: {all_nodes}\n"
            f"Relationships: ```{list_of_relationships}```\n"
            f"Information about the nodes:\n{nodes_info_str}\n"
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

def run_ablation_study(user_query: str, llm_config: Dict, schema: Dict, instructions_path: str) -> Dict[str, str]:
    """
    Runs an ablation study with different configurations.

    Args:
        user_query (str): The user's query.
        llm_config (Dict): The LLM configuration.
        schema (Dict): The schema information.
        instructions_path (str): Path to the instructions file.

    Returns:
        Dict[str, str]: Dictionary with results for each configuration.
    """
    results = {}

    # 1. With instructions (baseline)
    results["with_instructions"] = generate_with_instructions(user_query, llm_config, schema, instructions_path)
    
    # 2. Without instructions
    results["without_instructions"] = generate_without_instructions(user_query, llm_config, schema, instructions_path)
    
    # 3. Without schema comments
    results["without_schema_comments"] = generate_without_schema_comments(user_query, llm_config, schema, instructions_path)
    
    # 4. Without relevant nodes (use all nodes)
    results["without_relevant_nodes"] = generate_without_relevant_nodes(user_query, llm_config, schema, instructions_path)

    return results

def save_ablation_results(ablation_results: List[Dict[str, Any]], kg_name: str, model_name: str):
    """
    Saves ablation study results to a JSON file.

    Args:
        ablation_results (List[Dict[str, Any]]): The ablation study results.
        kg_name (str): The knowledge graph name.
        model_name (str): The model name.

    Returns:
        str: The path to the saved file.
    """
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results_path = f"./results/{kg_name}/"
    save_path = f"{results_path}ablation-{model_name}-{timestamp}.json"
    os.makedirs(results_path, exist_ok=True)
    with open(save_path, "w") as outfile:
        json.dump(ablation_results, outfile, indent=4)
    return save_path

def clean_cypher_query(query: str) -> str:
    """
    Cleans the Cypher query by removing unnecessary characters.

    Args:
        query (str): The Cypher query to clean.

    Returns:
        str: The cleaned Cypher query.
    """
    match_index = query.find("MATCH")
    if match_index != -1:
        query = query[match_index:]
    else:
        return query.strip()
    
    last_semicolon_index = query.rfind(";")
    if last_semicolon_index != -1:
        query = query[:last_semicolon_index + 1]
    
    query = query.replace("`", "")
    query = query.replace("\n", " ")
    return query.strip()

if __name__ == "__main__":
    import os
    import json
    from helpers.schema_utils import load_schema 
    from dotenv import load_dotenv

    load_dotenv()
    NEO4J_DATABASE_NAME = os.getenv("NEO4J_DATABASE_NAME")
    KG_NAME = None
    if NEO4J_DATABASE_NAME == "neo4j":
        KG_NAME = "ionchannels"
    elif NEO4J_DATABASE_NAME == "prokino-kg":
        KG_NAME = "prokino"
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # OpenAI model config
    MODEL_INFO = {"name": "gpt-4o-mini", "filename": "gpt-4o-mini", "litemodel": "gpt-4o-mini", "is_local": False}
    
    llm_config = {"config_list": [{"model": MODEL_INFO["name"], "litemodel": MODEL_INFO["litemodel"], "api_key": os.getenv("OPENAI_API_KEY")}]}

    # File paths
    TEST_QUERIES_PATH = f"./data/{KG_NAME}/test_queries.json"
    SCHEMA_PATH = f"./data/{KG_NAME}/schema.json"
    SCHEMA_COMMENTS_PATH = f"./data/{KG_NAME}/schema_comments.json"
    INSTRUCTIONS_PATH = f"./prompts/{KG_NAME}_instructions.txt"
    RESULTS_PATH = f"./results/{KG_NAME}/"
    
    # Load schema and query data
    schema, schema_comments = load_schema(SCHEMA_PATH, SCHEMA_COMMENTS_PATH)
    schema["nodes_description"] = schema_comments['nodes_description']
    schema["nodes_properties"] = schema_comments['nodes_properties']
    schema["relationship_comments"] = schema_comments['relationship_comments']
    
    with open(TEST_QUERIES_PATH, 'r') as file:
        test_queries = json.load(file)
    
    # Run ablation study for each query
    ablation_results = []
    
    print(f"Running ablation study on {len(test_queries)} queries...")
    for i, item in enumerate(test_queries):
        user_query = item["query"]
        title = item["title"]
        print(f"[{i+1}/{len(test_queries)}] Running query: {title}")
        
        # Run ablation study for this query
        results = run_ablation_study(user_query, llm_config, schema, INSTRUCTIONS_PATH)
        
        # Clean results
        for key in results:
            results[key] = clean_cypher_query(results[key])
        
        # Store results
        ablation_entry = {
            "title": title,
            "query": user_query,
            "ablation_results": results
        }
        ablation_results.append(ablation_entry)
    
    # Save results
    save_path = save_ablation_results(ablation_results, KG_NAME, MODEL_INFO["filename"])
    print(f"Ablation study results saved to {save_path}")