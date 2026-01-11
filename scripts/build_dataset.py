# scripts/build_dataset.py
import os, glob
import pandas as pd

def _decode_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].str.decode("utf-8")
            except Exception:
                pass
    return df

def merge_all_nhanes_data(root_folder: str) -> pd.DataFrame:
    pattern = os.path.join(root_folder, "*.[Xx][Pp][Tt]")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No XPT found in {root_folder}")

    demo_path = next((f for f in all_files if "DEMO" in os.path.basename(f).upper()), None)
    if not demo_path:
        raise FileNotFoundError("No DEMO*.XPT found (required as master).")

    master_df = _decode_object_cols(pd.read_sas(demo_path))
    all_files.remove(demo_path)

    for file_path in all_files:
        temp_df = _decode_object_cols(pd.read_sas(file_path))
        if "SEQN" in temp_df.columns and "SEQN" in master_df.columns:
            master_df = pd.merge(master_df, temp_df, on="SEQN", how="left")

    return master_df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-folder", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    df = merge_all_nhanes_data(args.data_folder)
    out = args.out_csv or os.path.join(args.data_folder, "NHANES_Merged_Data.csv")
    df.to_csv(out, index=False)
    print("✅ saved:", out, "| shape:", df.shape)
