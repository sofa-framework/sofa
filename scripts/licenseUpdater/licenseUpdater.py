import os
import glob
import argparse
import sys

# ANSI escape codes for colored output
RED = "\033[31m"
ORANGE = "\033[38;5;214m"
GREEN = "\033[32m"
RESET = "\033[0m"  # Reset color

def print_error(s):
    """Prints a string with [ERROR] in red"""
    print(f"{RED}[ERROR]{RESET} {s}")

def print_warning(s):
    """Prints a string with [WARNING] in orange"""
    print(f"{ORANGE}[WARNING]{RESET} {s}")

def print_info(s):
    """Prints a string with [INFO] in green"""
    print(f"{GREEN}[INFO]{RESET} {s}")

all_header_files = {
    'GPL' : "GPL_header.template",
    'LGPL': "LGPL_header.template"
}

all_headers = {}

def read_target_header(script_path : str, header_file_name : str):
    # Construct the path to the header file
    header_file_path = os.path.join(script_path, header_file_name)

    # Check if the header file exists
    if not os.path.exists(header_file_path):
        raise FileNotFoundError(f"The header file '{header_file_name}' was not found in the script's directory: {script_path}")

    # Read the content of the header file
    with open(header_file_path, 'r', encoding='utf-8') as header_file:
        target_header = header_file.read().strip()  # Use strip() to remove any leading/trailing whitespace

    return target_header

wrong_header_files = []

def add_header_if_missing(folder_path : str, extensions : list[str], selected_license : str, other_licenses : list[str], overwrite_wrong_headers : bool = False):
    if selected_license not in all_headers.keys():
        print_error(f"Cannot find the header for selected license '{selected_license}'")
        return

    target_header = all_headers[selected_license]

    # Iterate over each extension
    for ext in extensions:
        # Recursively find all files with the current extension in the folder
        for file_path in glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True):
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            has_wrong_header = False
            for license in other_licenses:
                if all_headers[license] in content:
                    print_warning(f"File {file_path} has a wrong header. The header for license {license} was found instead of {selected_license}.")
                    wrong_header_files.append(file_path)
                    has_wrong_header = True
                    break

            if has_wrong_header and not overwrite_wrong_headers:
                continue



            start_char = "/******************************************************************************"
            end_char = "******************************************************************************/"
            start_index = content.find(start_char)
            end_index = content.find(end_char, start_index + len(start_char))
            if start_index != -1 and end_index != -1:

                if content[start_index:end_index + len(end_char)] != target_header:

                    if has_wrong_header and overwrite_wrong_headers:
                        print_info(f"Overwriting wrong header in {file_path}")

                    new_content = target_header + content[end_index + len(end_char):]

                    # Write the modified content back to the file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)

                    print_info(f"Modified header to {file_path}")
            else:
                # Prepend the target header to the content
                new_content = target_header + '\n' + content

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)

                print_info(f"Added header to {file_path}")



if __name__ == "__main__":
    # Set up the argument parser with help messages
    parser = argparse.ArgumentParser(
        description="Add a specific header to C/C++ files if missing.",
        add_help=True)
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the files to process.")
    parser.add_argument('-l', '--license', type=str, help=f"Type of license to apply {list(all_header_files.keys())}", required=True)
    parser.add_argument('-o', '--overwrite', type=bool, help=f"Overwrite files with wrong headers. Default: False.", default=False)

    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        print_error(f"Folder '{args.folder_path}' does not exist.")
        sys.exit(1)

    args.folder_path = os.path.abspath(args.folder_path)

    if args.license not in all_header_files.keys():
        print_error(f"License '{args.license}' not supported. Supported licenses are {list(all_header_files.keys())}")
        sys.exit(1)

    selected_license = args.license
    print_info(f"License to apply: {selected_license}")

    if args.overwrite:
        print_info("Overwrite mode enabled.")

    # The header to apply is stored in this file
    target_header_file = all_header_files[args.license]

    # Get the script's path
    script_path = os.path.dirname(sys.argv[0])

    # Read the target header from the header file
    try:
        target_header = read_target_header(script_path, target_header_file)
        all_headers[selected_license] = target_header
    except FileNotFoundError as e:
        print_error(e)
        sys.exit(1)

    other_licenses = []
    for license_type, header_file in all_header_files.items():
        if license_type != selected_license:
            try:
                header = read_target_header(script_path, header_file)
                other_licenses.append(license_type)
                all_headers[license_type] = header
            except FileNotFoundError as e:
                print_warning(e)


    # Define the list of extensions to check
    extensions = ['h', 'cpp', 'inl', 'h.in', '.cu']

    add_header_if_missing(args.folder_path, extensions, selected_license, other_licenses, args.overwrite)

    print_warning(f"{len(wrong_header_files)} files with wrong header")
    for file_path in wrong_header_files:
        print(f"- {file_path}")
