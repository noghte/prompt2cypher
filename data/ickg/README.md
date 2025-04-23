# ICKG (Ion Channels Knowledge Graph)

## Dataset for Prompt2Cypher

This directory contains the ICKG (Ion Channels Knowledge Graph) dataset.

## Files

- `schema.json`: Graph schema definition with node labels (entities) and their properties, plus relationship types
- `schema_comments.json`: Annotations and descriptions for schema elements (it is not necessary to annotate the whole schema)
- `test_queries.json`: Collection of natural language queries for testing with their corresponding Cypher queries
- `test_queries-with_results.json`: Test queries along with their execution results

## Dataset Overview

ICKG is a specialized knowledge graph focused on ion channels and related biomedical information. We developed the ICKG by integrating data from repositories such as UniProt, STRING, The Gene Ontology, and Reactome. The graph comprises:

- 1,064,666 nodes of 18 distinct types (Protein, Pathway, Functional Domain, etc.)
- 6,288,620 edges of 18 relationship types (connecting proteins including ion channels to biological processes, diseases, ions, PTMs, etc.)

The knowledge graph was constructed using:
- 429 curated ion channel protein identifiers from UniProt
- Protein-protein interaction (PPI) data from STRING (filtered for high-confidence scores ≥700), adding 5,062 PPI edges
- 12,542 gene ontology relationship edges from The Gene Ontology database
- 296 reaction relationships from Reactome
- Manually curated relationships for edge types like hasGroup, hasFamily, hasUnit, hasGateMechanism, and hasIonAsso from ion channel-specific classifications

## Query Examples

The test queries are organized by complexity level:
- Level A: Simple, single-hop queries
- Level B: Medium complexity, two-hop queries
- Level C: Complex, multi-hop queries

## Usage

These files are used by the Prompt2Cypher system to benchmark language model performance in generating accurate Cypher queries for biomedical knowledge graph queries.

## Neo4j Installation and Configuration

1. [Download ICKG data](https://outlookuga-my.sharepoint.com/:u:/g/personal/ss44253_uga_edu/EU2cmyn6TsFEmA4ajU4YVg4BiY2Sl3DiQjmEeRv-EPpyMQ?e=5Pmma4)
2. Install Neo4j: https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04
3. In settings, change the heap size to a larger value, based on the computer's memory. For example:
    ```
    dbms.memory.heap.initial_size=4G
    dbms.memory.heap.max_size=16G
    ```

## Importing the CSV data into Neo4j using `neo4j-admin`

This method is the fastest, but the `neo4j-admin` tool likely doesn't work with Neo4j Desktop. While the Neo4j license allows for one graph database, there are workarounds to have multiple databases.

1. Make sure the CSV format is similar to the following:

`nodes.csv`:
```csv
id:ID,name,type:LABEL
0,A0A1D5NZX9,Protein
1,2422406,Pathway
2,433794,Pathway
3,437987,Pathway
```

`relationships.csv`:
```csv
:START_ID,:END_ID,:TYPE
0,1,hasPathway
0,2,hasPathway
0,3,hasPathway
4,5,hasPathway
```
2. Run `validate_data.py` and show possible errors, and fix them.

3. Run: 
 - `sudo systemctl stop neo4j`
 - Navigate to the data directory
 - If the license does not allow more than one database (default): `sudo neo4j-admin database import full --nodes=df_nodes_final.csv --relationships=df_edges_final.csv --overwrite-destination=true --verbose`
    
    | Otherwise you can run the same command with the database name at the end`sudo neo4j-adm.... --verbose ionchannels`. A workaround is to change default database by editing `/etc/neo4j/neo4j.conf`.
  
 - `sudo systemctl start neo4j`
 - `sudo neo4j restart`

**Troubleshooting**:
  - To delete the database, first stop neo4j, then: `rm -r /var/lib/neo4j/data/databases/ionchannels/`
  - To manually creating the database in `cypher-shell -u neo4j` run `CREATE DATABASE ionchannels`

