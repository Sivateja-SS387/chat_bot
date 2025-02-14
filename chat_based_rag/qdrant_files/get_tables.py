import psycopg2

# Connection parameters
connection_parameters = {
    'host': 'localhost',
    'port': '5434',
    'database': 'mms_dbs',
    'user': 'postgres',
    'password': 'admin'
}

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**connection_parameters)
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    
    print("\nTables in mms_dbs database:")
    print("------------------------")
    for table in tables:
        print(f"- {table[0]}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
