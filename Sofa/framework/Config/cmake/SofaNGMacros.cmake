include_guard(GLOBAL)

macro(sofang_add_component_subdirectory DirectoryArg FullDirectoryArg)
    set(DirectoryName ${FullDirectoryArg})

    string(TOUPPER ${DirectoryName} UpperDirectoryName)
    string(REPLACE "." "_" UpperDirectoryName ${UpperDirectoryName})

    option(SOFANG_ENABLE_${UpperDirectoryName} "Build ${DirectoryName}." ON)
    if(SOFANG_ENABLE_${UpperDirectoryName})
        add_subdirectory(${DirectoryArg})
    endif()
endmacro()
