# ProKino Knowledge Graph

## Dataset for Prompt2Cypher

This directory contains the ProKino (Protein Kinase Knowledge Graph) dataset used for evaluating natural language to Cypher query generation in the Prompt2Cypher project.

## Files

- `schema.json`: Graph schema definition with node labels (entities) and their properties, plus relationship types
- `schema_comments.json`: Annotations and descriptions for schema elements (it is not necessary to annonate the whole schema)
- `test_queries.json`: Collection of natural language queries for testing with their corresponding Cypher queries
- `test_queries-with_results.json`: Test queries along with their execution results

## Dataset Overview

ProKino is a specialized knowledge graph focused on protein kinases and their interactions. The schema includes various biological entities such as:

- Proteins and their structural/functional properties
- Mutations and their effects
- Organisms and taxonomic classifications
- Cancer types and samples
- Pathways and reactions
- Ligands and their interactions

The graph is particularly rich in relationships between these entities, allowing for complex queries across multiple biological domains.

## Query Examples

The test queries are organized by complexity level:
- Level A: Simple, single-hop queries (e.g., "What is the functional domain associated with EGFR protein?")
- Level B: Medium complexity, two-hop queries (e.g., "What are the sequence motifs found in the amino acid sequence of the EGFR protein?")
- Level C: Complex, multi-hop queries (e.g., "What ligands interact with the sequence motifs of the IRAK1 protein, and what is the type of interaction?")

## Usage

These files are used by the Prompt2Cypher system to benchmark language model performance in generating accurate Cypher queries for biological knowledge graph queries.

## Neo4j Installation and Configuration

1. Download [ProKinO's owl file](https://prokino.uga.edu/downloads/)
1. Install Neo4j: https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04
1. In settings, change the heap size to a larger value, based on the computer's memory. For example:
    ```
    dbms.memory.heap.initial_size=4G
    dbms.memory.heap.max_size=16G
    ```

## Importing the OWL ontology into Neo4j
- In Neo4j Desktop, add a local DBMS (verion `5.14`)
- Click on three dots beside the DBMS name, open folder, plugins.
- Copy the `plugins/neosemantics-5.14.0.jar` from this repo to the plugin folder.
- Start the DMBS.
- Create a database (e.g., `prokino-kg`).
- Open the DBMS and run
    1. ```cypher
        CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE
        ```
    2.  Run the following command to initialize the graph in Neo4j. Pay attention to the `handleVocabUris` and `handleMultival` parameters. [More info](https://neo4j.com/labs/neosemantics/5.14/reference/):
        ```cypher
        CALL n10s.graphconfig.init({
        handleVocabUris: 'IGNORE', 
        handleMultival: 'OVERWRITE',
        multivalPropList: [''], //provide a list of properties that are arrays (full uris)
        handleRDFTypes: 'LABELS_AND_NODES',
        keepLangTag: false
            })
        ```
        Older version:
        ```cypher
        CALL n10s.graphconfig.init({
        handleVocabUris: 'IGNORE', 
        handleMultival: 'ARRAY',
        handleRDFTypes: 'LABELS_AND_NODES',
        keepLangTag: true
            })
        ```
    3. `CALL n10s.rdf.import.fetch("file:///path/to/Prokino_v62_2021-09-15.owl","RDF/XML");` it might take 1 to 3 hours to import the OWL file of `58,636,829` triples from this version.

## Postprocessing

Set isDarkKinase to false if this property is missing:
```cypher
MATCH (p:Protein)
WHERE p.isDarkKinase IS NULL
SET p.isDarkKinase = false
RETURN count(p) AS UpdatedNodes
```

Remove Resource labels if they are not needed:
```cypher
MATCH (n)
WHERE "Resource" IN labels(n) AND size(labels(n)) > 1
REMOVE n:Resource
RETURN count(n) AS UpdatedNodes
```