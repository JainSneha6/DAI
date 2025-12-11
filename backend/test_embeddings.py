import psycopg2, binascii, json, numpy as np

pg_uri = "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"
conn = psycopg2.connect(pg_uri)
cur = conn.cursor()

# 1) get columns
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='cyborg_items' ORDER BY ordinal_position;")
cols = cur.fetchall()
print("cyborg_items columns:", cols)

# 2) fetch rows
cur.execute("SELECT * FROM cyborg_items LIMIT 5;")
rows = cur.fetchall()
col_names = [c[0] for c in cur.description]
print("col_names:", col_names)
for r in rows:
    print("--- row ---")
    for name, val in zip(col_names, r):
        print(name, "->", type(val))
        if isinstance(val, memoryview) or isinstance(val, (bytes, bytearray)):
            b = bytes(val)
            print("  length:", len(b))
            # show short hex preview
            print("  hex_preview:", binascii.hexlify(b)[:120])
            # try decode as utf-8 (if text)
            try:
                text = b.decode('utf-8')
                print("  utf8:", text[:400])
            except Exception:
                print("  utf8: not valid text")
            # try interpret as float32 array (if length divisible by 4)
            if len(b) % 4 == 0 and len(b) >= 4:
                try:
                    arr = np.frombuffer(b, dtype=np.float32)
                    print("  float32 array shape:", arr.shape, "preview:", arr[:8])
                except Exception:
                    pass
        else:
            print("  value:", val)

cur.close()
conn.close()
