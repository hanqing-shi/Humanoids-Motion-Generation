#!/usr/bin/env python3
"""
Analyze joystick log CSV to find which bytes correspond to left/right sticks.
Looks for continuous value changes (analog sticks) vs discrete values (D-pad).
"""

import csv
import sys
from pathlib import Path
import numpy as np

def analyze_csv(csv_file):
    """Analyze CSV file to find analog stick mappings."""
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    print(f"Analyzing: {csv_file}\n")
    print("=" * 80)
    
    # Read CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) == 0:
        print("❌ No data in CSV file")
        return
    
    print(f"Total samples: {len(rows)}\n")
    
    # Get all byte columns
    byte_cols = [col for col in rows[0].keys() if col.startswith('byte_')]
    byte_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Analyze each byte for continuity
    print("Byte Analysis (looking for continuous analog values):")
    print("-" * 80)
    print(f"{'Byte':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'Unique':<8} {'StdDev':<8} {'Type':<15}")
    print("-" * 80)
    
    byte_stats = []
    for byte_col in byte_cols:
        values = [int(row[byte_col]) for row in rows if row[byte_col]]
        if not values:
            continue
        
        values = np.array(values)
        unique_count = len(np.unique(values))
        std_dev = np.std(values)
        value_range = np.max(values) - np.min(values)
        
        # Determine type
        if unique_count < 10 and value_range < 10:
            byte_type = "D-pad/Button"
        elif unique_count > 50 and value_range > 100:
            byte_type = "Analog (likely)"
        elif unique_count > 20:
            byte_type = "Analog (maybe)"
        else:
            byte_type = "Unknown"
        
        byte_stats.append({
            'byte': byte_col,
            'min': np.min(values),
            'max': np.max(values),
            'range': value_range,
            'unique': unique_count,
            'stddev': std_dev,
            'type': byte_type,
            'values': values
        })
        
        print(f"{byte_col:<8} {np.min(values):<8} {np.max(values):<8} {value_range:<8} "
              f"{unique_count:<8} {std_dev:<8.2f} {byte_type:<15}")
    
    print("\n" + "=" * 80)
    print("Mapping Analysis:")
    print("-" * 80)
    
    # Analyze mappings
    mapping_cols = [col for col in rows[0].keys() if 'map' in col and 'axis' in col]
    mapping_groups = {}
    for col in mapping_cols:
        map_name = col.split('_axis')[0]
        if map_name not in mapping_groups:
            mapping_groups[map_name] = []
        mapping_groups[map_name].append(col)
    
    for map_name, cols in mapping_groups.items():
        print(f"\n{map_name}:")
        for col in sorted(cols):
            values = [float(row[col]) for row in rows if row[col]]
            if values:
                values = np.array(values)
                unique_count = len(np.unique(values))
                std_dev = np.std(values)
                value_range = np.max(values) - np.min(values)
                
                continuity = "✅ Continuous" if unique_count > 50 and std_dev > 0.1 else "❌ Discrete/Jumpy"
                
                print(f"  {col:<25} unique={unique_count:>4}, std={std_dev:>6.3f}, range={value_range:>6.3f} {continuity}")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("-" * 80)
    
    # Find best candidates for analog sticks
    analog_candidates = [b for b in byte_stats if b['type'].startswith('Analog')]
    analog_candidates.sort(key=lambda x: x['unique'], reverse=True)
    
    if len(analog_candidates) >= 4:
        print("\nTop analog byte candidates (likely stick positions):")
        for i, candidate in enumerate(analog_candidates[:8]):
            print(f"  {candidate['byte']}: {candidate['unique']} unique values, "
                  f"range={candidate['range']}, std={candidate['stddev']:.2f}")
        
        print("\nSuggested mapping:")
        print(f"  Left Stick X:  {analog_candidates[0]['byte']}")
        print(f"  Left Stick Y:  {analog_candidates[1]['byte']}")
        print(f"  Right Stick X: {analog_candidates[2]['byte']}")
        print(f"  Right Stick Y: {analog_candidates[3]['byte']}")
    
    # Check which mapping has continuous values
    print("\nMapping continuity check:")
    for map_name in sorted(mapping_groups.keys()):
        axis_cols = [f"{map_name}_axis{i}" for i in range(4)]
        all_continuous = True
        for col in axis_cols:
            if col in [c for group in mapping_groups.values() for c in group]:
                values = [float(row[col]) for row in rows if row[col]]
                if values:
                    unique_count = len(np.unique(values))
                    if unique_count < 50:
                        all_continuous = False
                        break
        
        status = "✅ All continuous" if all_continuous else "❌ Has discrete values"
        print(f"  {map_name}: {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest CSV file
        csv_dir = Path(__file__).parent
        csv_files = sorted(csv_dir.glob("joystick_test_*.csv"), reverse=True)
        if csv_files:
            csv_file = csv_files[0]
            print(f"Using latest log file: {csv_file.name}\n")
        else:
            print("Usage: python analyze_joystick_log.py <csv_file>")
            print("Or place joystick_test_*.csv in the same directory")
            sys.exit(1)
    else:
        csv_file = sys.argv[1]
    
    analyze_csv(csv_file)

