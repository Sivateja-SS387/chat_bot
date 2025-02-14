import psycopg2

connection_parameters = {
    'host': 'localhost',
    'port': '5433',
    'database': 'mms_dbs',
    'user': 'postgres',
    'password': 'admin'
}

try:
    conn = psycopg2.connect(**connection_parameters)
    cur = conn.cursor()
    
    # Get columns for drug_labels table
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'drug_labels'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()
    
    print("\nColumns in drug_labels table:")
    print("---------------------------")
    for col in columns:
        print(f"- {col[0]} ({col[1]})")
    
    # Get a sample row
    cur.execute("""
        SELECT * FROM drug_labels 
        LIMIT 1;
    """)
    sample = cur.fetchone()
    if sample:
        print("\nSample row:")
        print("-----------")
        for col, val in zip([c[0] for c in columns], sample):
            print(f"{col}: {val}")
    
    # Get count of rows
    cur.execute("SELECT COUNT(*) FROM drug_labels")
    count = cur.fetchone()[0]
    print(f"\nTotal rows in drug_labels: {count}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
