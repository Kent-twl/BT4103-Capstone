import os
import sys
from sqlalchemy import create_engine, MetaData, Table, text
import pandas as pd
from pprint import pprint

ROOT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.dirname(__file__))

DATA_DIR = ROOT_DIR + "/data/raw"

## Creating the orders database
def create_orders_db(source_file, dest_dir, show_samples=False):
    assert source_file.endswith('.csv') or source_file.endswith('.xlsx'), \
        "Data file must be in CSV or Excel format"
    
    df = pd.read_csv(source_file) if source_file.endswith('.csv') else pd.read_excel(source_file)
    sqlite_db_path = dest_dir + "/orders.db"
    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    with engine.connect() as conn:
        print(f"Creating orders database...")
        rows = df.to_sql(name="orders", con=engine, if_exists="replace", index=False)
        print(f"Inserted {rows} rows into the orders table")

        ## Check database
        table = Table('orders', MetaData(), autoload_with=engine)
        print(f"Columns in table '{table.name}':")
        pprint(table.columns.values())

        rows = conn.execute(text("SELECT * FROM orders LIMIT 5")).fetchall()
        if show_samples:
            print(f"Sample rows in table '{table.name}':")
            for row in rows:
                print(row)

    engine.dispose()

    print("Database created successfully!")


if __name__ == "__main__":
    create_orders_db(source_file=DATA_DIR+"/anonymised_data_v3.csv", dest_dir=DATA_DIR, show_samples=True)