import pandas as pd
import os

def clean_rollout_csvs(input_folder):
    visualize_dir = os.path.join(input_folder, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)

    # Loop through all CSVs in the folder
    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue
        if "clean" in file or "visualize" in file:
            continue  # skip already processed or visualize folder outputs

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(visualize_dir, file)

        print(f"🔍 Processing: {file}")

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"⚠️ Skipping {file} (read error: {e})")
            continue

        # Identify non-configuration columns
        drop_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["frame", "time", "step", "loss", "phase"]):
                drop_cols.append(col)

        if drop_cols:
            print(f"🧩 Dropping columns: {drop_cols}")
            df = df.drop(columns=drop_cols)

        # Convert everything to numeric, drop bad rows
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Save without headers/index
        df.to_csv(output_path, index=False, header=False)
        print(f"✅ Saved cleaned file → {output_path}")

    print("\n🎉 Done! All cleaned CSVs are now in:")
    print(f"   {visualize_dir}")
    print("You can now run rerun_visualize.py on them, e.g.:\n"
          f"   python rerun_visualize.py --file_name visualize/<filename> --robot_type g1")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_clean_rollouts.py <folder_path>")
        sys.exit(1)

    input_folder = sys.argv[1]
    clean_rollout_csvs(input_folder)
