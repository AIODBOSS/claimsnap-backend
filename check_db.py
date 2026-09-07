import sqlite3

for dbfile in ["instance/claimsnap.db", "instance/memory_layer.db"]:
    print(f"\n--- {dbfile} ---")
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print(tables)
    conn.close()
