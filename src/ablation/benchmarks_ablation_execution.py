import json
import os
import subprocess
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node

# Load environment variables
load_dotenv()

# DEFAULT_ABLATION_FILE = "ablation-gpt-4o-mini-2025_04_01-17_02_40.json" #ProKinO
DEFAULT_ABLATION_FILE = "ablation-gpt-4o-mini-2025_04_08-13_49_22.json" #ICKG

parser = argparse.ArgumentParser(description="Process ablation study results")
parser.add_argument("--ablation_file", 
                    type=str,
                    default=DEFAULT_ABLATION_FILE,
                    help="The JSON file containing the ablation study results")

args = parser.parse_args()
ABLATION_FILE = args.ablation_file

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE_NAME = os.getenv("NEO4J_DATABASE_NAME")
KG_NAME = None
if NEO4J_DATABASE_NAME == "neo4j":
    KG_NAME = "ickg"
elif NEO4J_DATABASE_NAME == "prokino-kg":
    KG_NAME = "prokino"

input_file_path = f'./results/{KG_NAME}/{ABLATION_FILE}'
output_file_path = input_file_path.replace('.json', '-with_results.json')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clean_value(value):
    if isinstance(value, Node):
        # Convert Node object to a dictionary
        return dict(value.items())
    elif isinstance(value, str):
        return value.replace('\u00a0', ' ').strip()
    return value

def run_cypher_query(query):
    try:
        with driver.session(database=NEO4J_DATABASE_NAME) as session:
            if query:
                results = session.run(query)
                result_list = [[clean_value(value) for value in record.values()] for record in results]
                return result_list
            else:
                return []
    except Neo4jError as e:
        print(f"Neo4j query failed: {e}")
        return []

# Read the input JSON file
try:
    with open(input_file_path, 'r') as file:
        data = json.load(file)
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON file: {e}")
    exit(1)

# Process each item in the JSON data
for item in data:
    title = item.get("title", "Unknown")
    print(f"Processing query: {title}")
    
    # Check if the 'metrics' field is empty and remove it if it is
    if "metrics" in item and (not item["metrics"] or not isinstance(item["metrics"], dict) or len(item["metrics"]) == 0):
        del item["metrics"]
    
    ablation_results = item.get("ablation_results", {})
    
    # For each ablation type, execute the Cypher query
    for ablation_type, cypher_query in ablation_results.items():
        print(f"  Running {ablation_type}...")
        try:
            # Handle both string queries and dict objects with cypher_query field
            query_to_run = cypher_query
            if isinstance(cypher_query, dict) and "cypher_query" in cypher_query:
                query_to_run = cypher_query["cypher_query"]
            
            cypher_result = run_cypher_query(query_to_run)
            ablation_results[ablation_type] = {
                "cypher_query": query_to_run,
                "cypher_result": cypher_result
            }
        except Exception as e:
            print(f"  Error running query for {ablation_type}: {e}")
            query_str = cypher_query
            if isinstance(cypher_query, dict) and "cypher_query" in cypher_query:
                query_str = cypher_query["cypher_query"]
                
            ablation_results[ablation_type] = {
                "cypher_query": query_str,
                "cypher_result": [],
                "error": str(e)
            }
    
    # Update the item with the results
    item["ablation_results"] = ablation_results

# Write the updated data to the output JSON file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

# Close the Neo4j driver
driver.close()

print(f"Updated JSON file saved as {output_file_path}")

# Create a benchmark-style format for calculating metrics
def convert_to_benchmark_format(ablation_data):
    benchmark_format = []
    
    for item in ablation_data:
        title = item.get("title", "Unknown")
        query = item.get("query", "")
        ablation_results = item.get("ablation_results", {})
        
        benchmark_entry = {
            "title": title,
            "description": query,
            "results": {}
        }
        
        # Add each ablation type as a separate version
        for ablation_type, result in ablation_results.items():
            if isinstance(result, dict) and "cypher_query" in result:
                # Already in the right format with cypher_query and cypher_result
                benchmark_entry["results"][ablation_type] = {
                    "cypher_query": result.get("cypher_query", ""),
                    "cypher_result": result.get("cypher_result", [])
                }
            else:
                # Handle case where result is just a string (cypher query without results yet)
                benchmark_entry["results"][ablation_type] = {
                    "cypher_query": result if isinstance(result, str) else "",
                    "cypher_result": []
                }
        
        benchmark_format.append(benchmark_entry)
    
    return benchmark_format

# Convert and save in benchmark format for metrics calculation
benchmark_format_data = convert_to_benchmark_format(data)
benchmark_format_path = output_file_path.replace('-with_results.json', '-benchmark_format-with_results.json')

with open(benchmark_format_path, 'w') as file:
    json.dump(benchmark_format_data, file, indent=4)

print(f"Benchmark format saved as {benchmark_format_path}")
