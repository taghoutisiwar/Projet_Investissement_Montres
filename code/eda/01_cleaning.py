import os, importlib.util

HERE = os.path.abspath(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("setup_paths", os.path.join(HERE, "00_setup_paths.py"))
setup_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_paths)

def main():
    df_raw = setup_paths.load_data(setup_paths.RAW_CSV)
    df_clean = setup_paths.basic_cleaning(df_raw)

    # Écrit dans racine/data/watches_clean_v1.csv 
    df_clean.to_csv(setup_paths.CLEAN_CSV, index=False)
    print(f"OK -> {setup_paths.CLEAN_CSV}")

if __name__ == "__main__":
    main()