import os
import re
import argparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Global state management for allowed mesh paths.
ALLOWED_MESHES: Set[str] = set()


def collect_mesh_paths(mesh_dir: str) -> None:
    """
    Scans the specified mesh directory recursively and collects all unique filenames
    into the global ALLOWED_MESHES set. Paths are stored relative to mesh_dir.

    Args:
        mesh_dir: The root path containing all potential mesh assets.
    """
    global ALLOWED_MESHES
    ALLOWED_MESHES.clear()  # Reset state before collection

    abs_mesh_dir = os.path.abspath(mesh_dir)
    print(f"\n--- Phase 1/3: Collecting valid mesh assets from: {abs_mesh_dir} ---")

    if not os.path.isdir(abs_mesh_dir):
        print(f"[ERROR] Mesh Definition Directory not found at: {abs_mesh_dir}")
        return

    file_count = 0
    for root, _, files in os.walk(abs_mesh_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            # Store the path relative to the mesh directory and ensure forward slashes for consistency
            relative_path = os.path.relpath(full_path, start=abs_mesh_dir).replace('\\', '/')
            ALLOWED_MESHES.add(relative_path)
            file_count += 1

    print(f"[SUCCESS] Collected {len(ALLOWED_MESHES)} unique mesh assets from {file_count} files.")


def normalize_potential_reference(raw_match: str) -> str:
    """
    Cleans up and standardizes a raw path match found in the code line.

    Args:
        raw_match: The raw string segment matched from a project file (e.g., "mesh/assets/A.scn").

    Returns:
        The cleaned, normalized path segment. Returns an empty string if the match is discarded.
    """
    # 1. Normalize separators and strip leading/trailing slashes
    normalized = raw_match.replace('\\', '/').strip('/')

    # 2. Strip common 'mesh/' prefixes aggressively
    if normalized.lower().startswith('mesh/'):
        return normalized[len("mesh/"):]
    elif normalized.lower() == 'mesh':
        return ""  # Discard if only the word "mesh" is found

    return normalized


def find_mesh_references(project_dir: str, mesh_dir: str, delete_unused: bool) -> None:
    """
    Main scanning function. Scans project files to identify usage of defined meshes,
    reports usage locations (deterministically), and calculates unused asset size.

    Args:
        delete_unused: If True, attempts to delete identified unused assets after confirmation.
    """
    global ALLOWED_MESHES

    # --- Phase 1: Collect Meshes ---
    collect_mesh_paths(mesh_dir)

    if not ALLOWED_MESHES:
        print("\n[FATAL] Cannot proceed with scanning because no valid mesh assets were collected.")
        return

    # Dictionary to store {canonical_mesh_path: list[(filepath, lineno)]}
    unique_references: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    total_files_scanned = 0
    lines_processed = 0

    def process_file(filepath: str):
        """Reads a single file, extracts potential references, and validates them against allowed meshes."""
        nonlocal lines_processed
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for lineno, line in enumerate(f, 1):
                    lines_processed += 1

                    # Regex captures potential paths starting after 'mesh' variations.
                    matches = re.findall(r'(?:["\']mesh[\/\\])(.*?)(?:["\'])', line)

                    for raw_path in matches:
                        reference_key = normalize_potential_reference(raw_path)
                        if not reference_key: continue

                        # Check against all allowed canonical paths
                        for canonical_mesh_path in ALLOWED_MESHES:
                            found_match = False

                            # Exact Match
                            if reference_key == canonical_mesh_path:
                                unique_references[canonical_mesh_path].append((os.path.abspath(filepath), lineno))
                                found_match = True
                                break

        except Exception as e:
            print(f"  [!] Warning: Error reading file {os.path.basename(filepath)}. Skipping. Reason: {e}")

    # Traverse the project directory
    for root, _, files in os.walk(project_dir):
        for filename in files:
            if filename.endswith(('.xml', '.scn', '.php', '.py', '.pyscn', '.cpp', '.h', '.inl')):
                full_path = os.path.join(root, filename)
                process_file(full_path)
                total_files_scanned += 1

    # --- Phase 3: Summary Report Generation and Usage Check ---
    print("\n" + "=" * 70)
    print("--- Phase 3/3: Generating Report ---")
    print("-" * 30)
    print(f"Total project files scanned: {total_files_scanned}")
    print(f"Total lines processed (containing 'mesh'): {lines_processed}")
    print(f"\n[RESULT] Found unique, valid references matching defined meshes: {len(unique_references)}")
    print("=" * 70)

    # --- 3A. REPORTING USED ASSETS (Deterministic Order Ensured) ---
    sorted_keys = sorted(unique_references.keys())
    for i, reference_key in enumerate(sorted_keys):
        locations = unique_references[reference_key]
        print("\n" + "*" * 5)
        print(f">>> USED REFERENCE {i + 1}: {reference_key} <<<")
        print("*" * 5)

        # Use a set to find only unique location tuples (filepath, lineno)
        unique_locations = list(set(locations))

        # Sort the locations to ensure deterministic output order.
        unique_locations.sort()

        print(f"  Found at {len(unique_locations)} unique locations:")
        for filepath, lineno in unique_locations:
            relative_file_path = os.path.relpath(filepath)
            print(f"    - File: {relative_file_path}")
            print(f"      Line Number: {lineno}")

    # --- 3B. USAGE CHECK & SIZE REPORTING (Finding Unused Meshes) ---
    print("\n\n" + "=" * 70)
    print("--- Phase 3B: USAGE CHECK: FINDING UNUSED MESH ASSETS ---")
    print("-" * 30)

    used_references = set(unique_references.keys())
    initial_unused_references = ALLOWED_MESHES - used_references

    # Post-processing filter for MTL files (As per instructions)
    refined_unused_references: Set[str] = set()
    print("[INFO] Running companion asset analysis (MTL files)...")

    for mesh_path in initial_unused_references:
        is_mtl = mesh_path.lower().endswith('.mtl')

        if is_mtl:
            # If it's an MTL file, check if its presumed companion geometry exists and IS used.

            # 1. Get the directory path of the unused MTL file
            directory = os.path.dirname(mesh_path)
            base_name = os.path.basename(mesh_path).replace('.mtl', '')

            companion_found_and_used = False
            for allowed_asset in ALLOWED_MESHES:
                if os.path.dirname(allowed_asset) != directory and allowed_asset != mesh_path:
                    continue  # Must be in the same folder (or root if path is relative to nothing)

                # Check if this asset is a potential geometry holder AND is used
                is_geometry_holder = ('.obj' in allowed_asset.lower() or
                                      '.scn' in allowed_asset.lower() or
                                      '.dae' in allowed_asset.lower())

                if is_geometry_holder and allowed_asset in used_references:
                    # Found a geometry file (A.obj) that IS USED, so the MTL (B.mtl) might also be needed.
                    companion_found_and_used = True
                    break

            if companion_found_and_used:
                print(
                    f"  [INFO] Marked {mesh_path} as potentially used because its parent mesh was found in the project files.")
                continue  # Do not add this MTL file to the unused list

        # If it wasn't an MTL, or if it was an MTL and no used companion was found,
        # then it remains genuinely unused.
        refined_unused_references.add(mesh_path)

    unused_references = refined_unused_references  # Use the filtered set for reporting

    total_unused_size = 0

    if not unused_references:
        print("[SUCCESS] All collected mesh assets appear to be actively referenced or implicitly linked!")
    else:
        print(f"[WARNING] Found {len(unused_references)} potential UNUSED mesh asset(s).")

        # --- Deletion Execution Block ---
        if delete_unused:
            print("\n!!! WARNING !!! DELETION MODE ACTIVATED.")
            delete_confirm = input("Are you SURE you want to DELETE these unused assets? (y/N): ").lower()
            if delete_confirm != 'y':
                print("[ABORTED] Deletion cancelled by user.")
                return  # Exit early if deletion is refused

        # --- Reporting and Deleting Assets ---
        sorted_unused = sorted(list(unused_references))  # Ensure consistent order for unused assets

        for i, unused_key in enumerate(sorted_unused):
            full_path = os.path.join(os.path.abspath(mesh_dir), unused_key)
            file_size = 0
            try:
                file_size = os.path.getsize(full_path)
            except OSError as e:
                print(f"  -> {i + 1}. {unused_key} [Skipped: Could not read file size ({e})]")
                continue

            total_unused_size += file_size

            # Report the unused asset
            if os.path.exists(full_path):  # Check existence before printing/deleting
                print(f"  -> {i + 1}. {unused_key}")
                if unused_key.lower().endswith('.mtl'):
                    print("     [Note]: This material file might be redundant if its companion mesh is also unused.")
                print(f"     [Size]: {format_bytes(file_size)}")

                # Perform deletion if the flag is set and confirmed
                if delete_unused:
                    try:
                        os.remove(full_path)
                        print("     [ACTION] DELETED successfully.")
                    except OSError as e:
                        print(f"     [ERROR] Failed to delete {unused_key}: {e}")

        # Display Total Size (This sum should now also be deterministic)
        print("\n" + "=" * 70)
        if total_unused_size > 0:
            print("TOTAL UNUSED ASSET SIZE:")
            print(f"{format_bytes(total_unused_size)}")
        else:
            print("No measurable unused assets found.")
        print("=" * 70)


def format_bytes(size: int) -> str:
    """Converts a size in bytes into a human-readable string (KB, MB, GB)."""
    units = ['B', 'KB', 'MB', 'GB']
    s = float(size)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024.0
        i += 1
    return f"{s:.2f} {units[i]}"


def main():
    """
    Handles command line arguments and initiates the mesh reference scanning process.

    Usage: python script_name.py <project_dir> <mesh_definition_dir> [--delete-unused]
    """
    parser = argparse.ArgumentParser(
        description="Scans project files (XML, SCN, PY, PHP) to identify which defined mesh assets are actually used in the codebase. Helps detect unused/redundant assets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("project_directory", type=str,
                        help="The root folder path containing project source files to scan (e.g., 'src').")
    parser.add_argument("mesh_definition_directory", type=str,
                        help="The root folder path containing ALL mesh asset files used for validation (e.g., 'assets/meshes').")
    # New optional argument: --delete-unused or -d
    parser.add_argument("--delete-unused", "-d", action="store_true",
                        help="If provided, the script will delete all identified unused assets after asking for explicit confirmation.")

    args = parser.parse_args()
    find_mesh_references(args.project_directory, args.mesh_definition_directory, args.delete_unused)


if __name__ == "__main__":
    # Note: To test this script, ensure you run it with appropriate directory structures
    # (e.g., python your_script_name.py ./src ./assets/meshes --delete-unused)
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")

