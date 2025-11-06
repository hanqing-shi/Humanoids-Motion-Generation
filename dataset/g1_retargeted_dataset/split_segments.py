import csv
import os
import numpy as np
import pandas as pd

def extract_segments(segment_csv: str, input_folder: str, output_folder: str):
    """
    Extract specific row ranges from multiple CSV files based on a definition file.

    Args:
        segment_csv: Path to a CSV that defines segments (columns: file, start, end, out)
        input_folder: Folder containing the source CSV files
        output_folder: Folder to save the extracted segment CSVs
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(segment_csv)

    for _, row in df.iterrows():
        file_name = row['file']
        start = int(row['start'])
        end = int(row['end'])
        out_name = row['out']

        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, out_name + '.csv')

        if not os.path.exists(in_path):
            print(f"⚠️ Source file not found: {in_path}")
            continue

        # Read the source CSV
        data = np.genfromtxt(in_path, delimiter=',')

        # Sanity check for range
        if start < 0 or end > len(data):
            print(f"⚠️ Invalid range ({start}, {end}) for {file_name} (len={len(data)})")
            continue

        # Slice the desired segment
        segment = data[start:end, :]

        # Save as new CSV
        np.savetxt(out_path, segment, delimiter=',')
        print(f"✅ Saved {out_path} ({len(segment)} rows)")

    print("🎉 All segments processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split CSV files based on a segment definition file.")
    parser.add_argument('--segment_csv', type=str, default='./g1_retargeted_dataset/segments_hanqing_251015.csv',
                        help="Path to the segment definition CSV file (must contain file,start,end,out columns)")
    parser.add_argument('--input_folder', type=str, default='./g1_retargeted_dataset',
                        help="Folder containing the source CSV files")
    parser.add_argument('--output_folder', type=str, default='./g1_clipped_retargeted_dataset/walk',
                        help="Folder to save the output CSVs")
    args = parser.parse_args()

    extract_segments(args.segment_csv, args.input_folder, args.output_folder)
