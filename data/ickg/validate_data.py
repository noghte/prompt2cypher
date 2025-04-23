import csv

def validate_csv(filename):
    print(f"Checking file: {filename}")
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            for field in row:
                if '\n' in field or '\r' in field:
                    print(f"Newline character found in row {row_num}")
                    return False
    return True

# Validate the nodes CSV
print(validate_csv('kg/ionchan_24_05_29/preprocessed/df_nodes_final.csv'))

# Validate the edges CSV
print(validate_csv('kg/ionchan_24_05_29/preprocessed/df_edges_final.csv'))
