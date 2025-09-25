import os
import glob
import csv
import argparse

'''
CSV COMPILER HELPER SCRIPT

Takes in folder of csv files and compiles all data losslessly to one csv file sorting with new track_ids ensuring no isolated tracks are combined

Example command line argument: python3 /path_to_code.py -i /path_to_folder_of_csvs -o output.csv

Output should be a csv containing each tracks data

'''

# Map standard field names to allowed aliases
COLUMN_ALIASES = {
    'track id': ['track id', 'trackid', 'track_id', 'id'],
    'x': ['x', 'position_x', 'pos_x'],
    'y': ['y', 'position_y', 'pos_y'],
    't': ['t', 'frame', 'position_t', 'time']
}

def find_column(header_map, aliases):
    for alias in aliases:
        if alias in header_map:
            return header_map[alias]
    return None

def normalize_column_names(headers):
    return {h.lower().strip(): h for h in headers}

def is_numeric(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def main(input_folder, output_file):
    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    combined_rows = []
    output_headers = ['track id', 'x', 'y', 't', 'new_track_id']

    new_track_id = 0
    prev_track_id = None

    for file in csv_files:
        try:
            with open(file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                print(f"📄 Columns in '{file}': {reader.fieldnames}")
                header_map = normalize_column_names(reader.fieldnames)

                matched_columns = {}
                for key, aliases in COLUMN_ALIASES.items():
                    found = find_column(header_map, aliases)
                    if found:
                        matched_columns[key] = found
                    else:
                        print(f"⚠️ Missing column for '{key}' in {file}")
                        matched_columns = None
                        break

                if matched_columns:
                    for row in reader:
                        raw_track_id = row[matched_columns['track id']]

                        if not is_numeric(raw_track_id):
                            continue  # ❌ Skip non-numeric track ids

                        current_track_id = int(raw_track_id)

                        if current_track_id != prev_track_id:
                            new_track_id += 1
                            prev_track_id = current_track_id

                        new_row = {
                            'track id': current_track_id,
                            'x': row[matched_columns['x']],
                            'y': row[matched_columns['y']],
                            't': row[matched_columns['t']],
                            'new_track_id': new_track_id
                        }
                        combined_rows.append(new_row)
                else:
                    print(f"⚠️ Skipping '{file}' — couldn't map all required columns.")

        except Exception as e:
            print(f"❌ Error reading '{file}': {e}")

    if combined_rows:
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=output_headers)
                writer.writeheader()
                writer.writerows(combined_rows)
            print(f"✅ Combined data saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error writing output file: {e}")
    else:
        print("❗ No valid data found to combine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine tracking data and assign new_track_id each time the track id changes. Ignores non-numeric track ids.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the folder containing CSV files.")
    parser.add_argument("-o", "--output_file", required=True, help="Path to the output CSV file.")

    args = parser.parse_args()
    main(args.input_folder, args.output_file)
