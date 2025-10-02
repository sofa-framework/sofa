# Set Taskflow_FOUND to false initially
set(Taskflow_FOUND FALSE)

set(Taskflow_VERSION "3.10.0")

# Specify the Taskflow include directory and library
find_path(Taskflow_INCLUDE_DIR
    NAMES taskflow/taskflow.hpp
    HINTS "${TASKFLOW_ROOT}"
)

# If both the include directory are found, set Taskflow_FOUND to true
if(Taskflow_INCLUDE_DIR)
    set(Taskflow_FOUND TRUE)

    # Read the content of the taskflow.hpp file
    file(READ "${Taskflow_INCLUDE_DIR}/taskflow/taskflow.hpp" file_contents)

    # Use a regular expression to extract the version number
    string(REGEX MATCH "#define TF_VERSION [0-9]+" version_line "${file_contents}")

    # Extract the numeric part of the version
    string(REGEX REPLACE "#define TF_VERSION " "" version_number "${version_line}")

    # Split the version number into major, minor, and patch components
    # Taskflow uses a version format like 300900 for 3.9.0
    math(EXPR major_version "${version_number} / 100000")
    math(EXPR minor_version "(${version_number} % 100000) / 100")
    math(EXPR patch_version "${version_number} % 100")

    # Format the version as X.Y.Z
    set(formatted_version "${major_version}.${minor_version}.${patch_version}")

    # Set the output variable
    set(Taskflow_VERSION "${formatted_version}")
elseif(SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("taskflow NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching taskflow...")

    # If Taskflow is not found, use FetchContent to download and build it
    include(FetchContent)

    FetchContent_Declare(
        taskflow
        GIT_REPOSITORY https://github.com/taskflow/taskflow.git
        GIT_TAG        v${Taskflow_VERSION}
    )

    FetchContent_GetProperties(taskflow)
    if(NOT taskflow_POPULATED)
        FetchContent_Populate(taskflow)
    endif()

    set(Taskflow_FOUND TRUE)
    set(Taskflow_INCLUDE_DIR ${taskflow_SOURCE_DIR})
else()
    message(FATAL_ERROR "DEPENDENCY taskflow NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Define TASKFLOW_ROOT to a valid path to the headers of the library, or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()

# Handle REQUIRED and QUIET options
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Taskflow
    REQUIRED_VARS Taskflow_FOUND Taskflow_INCLUDE_DIR
    VERSION_VAR Taskflow_VERSION
)

mark_as_advanced(Taskflow_INCLUDE_DIR)
