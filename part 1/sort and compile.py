import os
import csv

# Directory containing the CSV files
directory = "C:/Users/Ibrahim/Desktop/csv/csv/WC"

# Output file name
output_file = "WC_combined_data.csv"

# List to hold all the rows of data
data = []

# Loop over all CSV files in the directory and add them to a list
files = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        files.append(filename)

# Sort the files by prefix and then by last number in ascending order
files.sort(key=lambda x: (float(x.split('_')[1]), int(x.split('_')[2][:-15])))


# Loop over the sorted file names and read in the second row of data
for filename in files:
    prefix = filename[:-15]
    with open(os.path.join(directory, filename)) as csv_file:
        reader = csv.reader(csv_file)
        # Skip the first row (header) and add the second row of data to the list,
        # with the prefix as the first column
        next(reader)  # skip header
        row = next(reader)  # get second row
        data.append([prefix] + row)


# Write the combined data to a new CSV file
with open(output_file, 'w', newline='') as output:
    writer = csv.writer(output)
    # Write the header row with the extra column
    writer.writerow(['File'] + data[0])
    # Write the data rows
    for row in data:
        writer.writerow(row[:1] + row[1:])
