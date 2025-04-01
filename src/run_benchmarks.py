import os
import time
import json
import datetime
import subprocess
from helpers.schema_utils import load_schema 
# from generate_cypher_baseline import generate_cypher_query as cypher_gen
from generate_cypher_p2c import generate_cypher_query as cypher_gen
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

# LOCAL
# MODEL_INFO = {"name":"mradermacher/CodeLlama-7b-CypherGen-GGUF", "is_local":True}
# MODEL_INFO = {"name":"lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "is_local":True}

#NVIDIA
# MODEL_INFO = {"name":"nvidia/llama-3.1-nemotron-70b-instruct", "is_local":False}

# OPEN AI
MODEL_INFO = {"name": "gpt-4o-mini", "filename": "gpt-4o-mini", "litemodel": "gpt-4o-mini",  "is_local": False}

llm_config = None
if MODEL_INFO["is_local"]:
    llm_config={"config_list": [
        {"model": MODEL_INFO["name"], 
         "base_url": "http://localhost:1234/v1", 
         "api_key": "lm-studio",
         "price" : [0, 0],
         }
         ]}
elif MODEL_INFO["name"].startswith("nvidia"): # NVIDIA
    llm_config={"config_list": [{"model": MODEL_INFO["name"], "base_url" :"https://integrate.api.nvidia.com/v1", "api_key": os.getenv("NVIDIA_API_KEY")}]}
else: # OpenAI
    llm_config = {"config_list": [{"model": MODEL_INFO["name"], "litemodel": MODEL_INFO["litemodel"], "api_key": os.getenv("OPENAI_API_KEY")}]}

TEST_QUERIES_PATH = f"./data/{KG_NAME}/test_queries.json"
SCHEMA_PATH = f"./data/{KG_NAME}/schema.json"
SCHEMA_COMMENTS_PATH = f"./data/{KG_NAME}/schema_comments.json"
INSTRUCTIONS_PATH = f"./prompts/{KG_NAME}_instructions.txt"
RESULTS_PATH = f"./results/{KG_NAME}/"


schema, schema_comments = load_schema(SCHEMA_PATH, SCHEMA_COMMENTS_PATH)
schema["nodes_description"] = schema_comments['nodes_description']
schema["nodes_properties"] = schema_comments['nodes_properties']
schema["relationship_comments"] = schema_comments['relationship_comments']

def clean_cypher_query(query: str) -> str:
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


def run_benchmark(generate_cypher_query_func, user_query):
    start_time = time.time()
    cypher_query = generate_cypher_query_func(user_query, llm_config, schema, INSTRUCTIONS_PATH)
    end_time = time.time()
    cypher_query = clean_cypher_query(cypher_query)
    print(cypher_query)
    return {
        "cypher_query": cypher_query,
        "time_taken": end_time - start_time,
    }

def save_benchmark_results(benchmark_results):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    save_path = f"{RESULTS_PATH}benchmark-{MODEL_INFO['filename']}-{timestamp}.json"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(save_path, "w") as outfile:
        json.dump(benchmark_results, outfile, indent=4)
    return save_path

if __name__ == "__main__":
    with open(TEST_QUERIES_PATH, 'r') as file:
        test_queries = json.load(file)
        ## uncomment the following lines to include only specific queries       
        # inclusion_titles = {"A4", "C4"}
        # test_queries = [item for item in test_queries if item["title"] in inclusion_titles]

    # You can add multiple versions of the generate_cypher_query function here
    versions = [
        ("version_1", cypher_gen),
        # ("version_2", another_cypher_gen),
    ]

    for version_name, generate_cypher_query_func in versions:
        benchmark_results = []

        for item in test_queries:
            user_query = item["query"]
            title = item["title"]
            print("Running query:", title)
            
            result = run_benchmark(generate_cypher_query_func, user_query)

            benchmark_entry = {
                "title": title,
                "description": user_query,
                "results": {
                    version_name: result,
                }
            }
            benchmark_results.append(benchmark_entry)
        
        save_path = save_benchmark_results(benchmark_results)
        print(f"Benchmark results saved to {save_path}.")

        run_script = input("Do you want to run the benchmarks_cypher_execution script? (Y/N): ").strip().lower()
        if run_script == 'y':
            subprocess.run(["python", "./src/benchmarks_cypher_execution.py", "--cypher_from_llm", save_path.split("/")[-1]])
