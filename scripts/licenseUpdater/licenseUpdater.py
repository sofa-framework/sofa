import os
import glob
import argparse
import sys

def read_target_header(script_path, header_file_name):
    # Construct the path to the header file
    header_file_path = os.path.join(script_path, header_file_name)

    # Check if the header file exists
    if not os.path.exists(header_file_path):
        raise FileNotFoundError(f"The header file '{header_file_name}' was not found in the script's directory: {script_path}")

    # Read the content of the header file
    with open(header_file_path, 'r', encoding='utf-8') as header_file:
        target_header = header_file.read().strip()  # Use strip() to remove any leading/trailing whitespace

    return target_header

def add_header_if_missing(folder_path, extensions, target_header, skipped_header):
    # Iterate over each extension
    for ext in extensions:
        # Recursively find all files with the current extension in the folder
        for file_path in glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True):
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if skipped_header in content:
                print(f"File {file_path} has a GPL license")
                continue

            start_char = "/******************************************************************************"
            end_char = "******************************************************************************/"
            start_index = content.find(start_char)
            end_index = content.find(end_char, start_index + len(start_char))
            if start_index != -1 and end_index != -1:

                if not content[start_index:end_index + len(end_char) - 1] != target_header:
                    new_content = target_header + content[end_index + len(end_char):]

                    # Write the modified content back to the file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)

                    print(f"Modified header to {file_path}")
            else:
                # Prepend the target header to the content
                new_content = target_header + '\n' + content

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)

                print(f"Added header to {file_path}")


if __name__ == "__main__":
    # Set up argument parser with help messages
    parser = argparse.ArgumentParser(description="Add a specific header to C/C++ files if missing.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the files to process.")

    args = parser.parse_args()

    # Get the script's path
    script_path = os.path.dirname(sys.argv[0])

    # Read the target header from the LGPL_header.template file
    try:
        lgpl_target_header = read_target_header(script_path, 'LGPL_header.template')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    gpl_target_header = read_target_header(script_path, 'GPL_header.template')

    # Define the list of extensions to check
    extensions = ['h', 'cpp', 'inl', 'h.in', '.cu']

    add_header_if_missing(args.folder_path, extensions, lgpl_target_header, gpl_target_header)
