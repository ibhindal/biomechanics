#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // Directory containing the CSV files
    string directory = "/path/to/csv/files";

    // Output file name
    string output_file = "combined_data.csv";

    // List to hold all the rows of data
    vector<vector<string>> data;

    // Loop over all CSV files in the directory and add them to a vector
    vector<string> files;
    for (const auto &entry : filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".csv") {
            files.push_back(entry.path().stem().string());
        }
    }

    // Sort the files by prefix and then by last number in ascending order
    sort(files.begin(), files.end(), [](const string &a, const string &b) {
        string prefix_a = a.substr(0, a.find('_', a.find('_') + 1));
        string prefix_b = b.substr(0, b.find('_', b.find('_') + 1));
        if (prefix_a != prefix_b) {
            return prefix_a < prefix_b;
        } else {
            int last_number_a = stoi(a.substr(a.find_last_of('_') + 1));
            int last_number_b = stoi(b.substr(b.find_last_of('_') + 1));
            return last_number_a < last_number_b;
        }
    });

    // Loop over the sorted file names and read in the data
    for (const auto &filename : files) {
        string prefix = filename.substr(0, filename.find('_', filename.find('_') + 1));
        ifstream csv_file(directory + "/" + filename + ".csv");
        string line;
        // Skip the header row
        getline(csv_file, line);
        // Add the data to the list, with the prefix as the first column
        while (getline(csv_file, line)) {
            vector<string> row = {prefix, line};
            data.push_back(row);
        }
    }

    // Write the combined data to a new CSV file
    ofstream output(output_file);
    // Write the header row with the extra column
    output << "File,";
    output << data[0][0] << endl;
    // Write the data rows
    for (int i = 0; i < data.size(); i++) {
        output << data[i][0] << ",";
        output << data[i][1] << endl;
    }
    output.close();

    return 0;
}
