import json
import os
def load_schema(schema_path, schema_comments_path):
    """
    Load the schema from a JSON file.

    :param schema_path: The file path of the schema JSON file.
    :return: The schema as a dictionary.
    """
    schema_comments = None
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if os.path.exists(schema_comments_path):
        with open(schema_comments_path, "r", encoding="utf-8") as f:
            schema_comments = json.load(f)

    return schema, schema_comments

def get_node_properties(schema, node_label):
    """
    Get the properties of a node based on the given schema and node label.

    :param schema: The schema containing the labels and their properties.
    :param node_label: The label of the node to retrieve properties for.
    :return: A list of properties for the specified node label.
    """
    if node_label in schema["labels"]:
        properties = list(schema["labels"][node_label].keys())
    else:
        properties = []
    return properties

def get_relationship_info(schema, relationship_name):
    """
    Get the source and destination nodes of a relationship based on the given schema and relationship name.

    :param schema: The schema containing the relationships and their properties.
    :param relationship_name: The name of the relationship to retrieve properties for.
    :return: A dictionary of the source and destination nodes for the specified relationship name.
    """
    properties = [e for e in schema["relationships"] if e["name"] == relationship_name][0]
    return properties
