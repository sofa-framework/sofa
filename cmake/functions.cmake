
# Declare an option. Just as the standard 'option' command, it creates
# a cache variable, but it also stores the list of options it created, as
# well as their default values
function(sofa_option name type default_value description)
    set(${name} "${default_value}" CACHE ${type} "${description}")
    set(SOFA_OPTION_LIST ${SOFA_OPTION_LIST} ${name} CACHE INTERNAL "List of cmake options")
    if(NOT DEFINED SOFA_OPTION_DEFAULT_VALUE_${name})
        set(SOFA_OPTION_DEFAULT_VALUE_${name} "${default_value}" CACHE INTERNAL "Default value for ${name}")
    endif()
endfunction()
set(SOFA_OPTION_LIST CACHE INTERNAL "List of cmake options") # Reset option list

# Write to a file the list of build options, declared with sofa_options(),
# that are not any more set to their default values
function(sofa_save_option_list filename)
    set(human_readable_list "")
    set(cmake_option_list "")
    foreach(option ${SOFA_OPTION_LIST})
        if(NOT "${${option}}" STREQUAL "${SOFA_OPTION_DEFAULT_VALUE_${option}}")
            set(cmake_option_list "${cmake_option_list} -D${option}=${${option}}")
            set(human_readable_list "${human_readable_list}- ${option} = ${${option}}\n")
        endif()
    endforeach()
    if("${human_readable_list}" STREQUAL "")
        set(human_readable_list "(none)\n")
    endif()
    file(WRITE "${SOFA_BUILD_DIR}/${filename}"
"Those options differ from their default values:

${human_readable_list}
You should obtain a similar configuration for a fresh build of SOFA with the
following command. (Beware: this only takes into account the options that were
declared with the 'sofa_option' function; e.g. CMAKE_BUILD_TYPE will not be listed)

cmake${cmake_option_list} .
")
    message(STATUS "The list of the options you changed was saved to: ${filename}")
endfunction()

# group files
macro(GroupFiles fileGroup topGroup baseDir)
    string(REPLACE "_" " " fileGroupName ${fileGroup})
    string(TOLOWER ${fileGroupName} fileGroupName)
    string(REGEX MATCHALL "([^ ]+)" fileGroupNameSplit ${fileGroupName})

    set(finalFileGroupName)
    foreach(fileGroupNameWord ${fileGroupNameSplit})
        string(SUBSTRING ${fileGroupNameWord} 0 1 firstLetter)
        string(SUBSTRING ${fileGroupNameWord} 1 -1 otherLetters)
        string(TOUPPER ${firstLetter} firstLetter)
        if(finalFileGroupName)
            set(finalFileGroupName "${finalFileGroupName} ")
        endif()
        set(finalFileGroupName "${finalFileGroupName}${firstLetter}${otherLetters}")
    endforeach()

    foreach(currentFile ${${fileGroup}})
        set(folder ${currentFile})
        get_filename_component(filename ${folder} NAME)
        string(REPLACE "${filename}" "" folder ${folder})
        set(groupName "${finalFileGroupName}")
        if(NOT baseDir STREQUAL "")
            get_filename_component(finalBaseDir "${baseDir}" ABSOLUTE)
            get_filename_component(folder "${folder}" ABSOLUTE)
            file(RELATIVE_PATH folder ${finalBaseDir} ${folder})
        endif()
        if(NOT folder STREQUAL "")
            string(REGEX REPLACE "/+$" "" baseFolder ${folder})
            string(REPLACE "/" "\\" baseFolder ${baseFolder})
            set(groupName "${groupName}\\${baseFolder}")
        endif()
        if(NOT topGroup STREQUAL "")
            set(groupName "${topGroup}\\${groupName}")
        endif()
        source_group("${groupName}" FILES ${currentFile})
    endforeach()
endmacro()

# make relative path for a set of files
macro(ToRelativePath outFiles fromDirectory inFiles)
    unset(tmpFiles)
    foreach(inFile ${inFiles})
        file(RELATIVE_PATH outFile "${fromDirectory}" "${inFile}")
        list(APPEND tmpFiles "${outFile}")
    endforeach()

    set(${outFiles} ${tmpFiles})
endmacro()

# gather files
macro(GatherProjectFiles files directories filter) # group)
    foreach(currentDirectory ${${directories}})
        file(GLOB pathes "${currentDirectory}/${filter}")
        foreach(currentPath ${pathes})
            file(RELATIVE_PATH currentFile "${CMAKE_CURRENT_BINARY_DIR}" "${currentPath}")
            list(APPEND ${files} "${currentFile}")
            #source_group("${${group}}${currentDirectory}" FILES ${currentFile})
        endforeach()
    endforeach()
endmacro()

# generate mocced headers from Qt4 moccable headers
macro(SOFA_QT4_WRAP_CPP outfiles )
    # get include dirs
    QT4_GET_MOC_FLAGS(moc_flags)
    #QT4_EXTRACT_OPTIONS(moc_files moc_options ${ARGN})

    set(defines)
    foreach(it ${GLOBAL_COMPILER_DEFINES})
        list(APPEND defines "-D${it}")
    endforeach()

    foreach(it ${ARGN})
        get_filename_component(it ${it} ABSOLUTE)
        QT4_MAKE_OUTPUT_FILE(${it} moc_ cpp outfile)
        QT4_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags}" "${defines}" "${moc_options}")
        set(${outfiles} ${${outfiles}} ${outfile})
    endforeach()
endmacro()

# generate .h files from .ui files
macro(SOFA_QT4_WRAP_UI outfiles)
    foreach(it ${ARGN})     # it = foo.ui
        get_filename_component(basename ${it} NAME_WE) # basename = foo
        get_filename_component(infile ${it} ABSOLUTE) # infile = /absolute/path/to/foo.ui
        set(outHeaderFile "${CMAKE_CURRENT_BINARY_DIR}/ui_${basename}.h")
        add_custom_command(OUTPUT ${outHeaderFile}
            COMMAND ${QT_UIC_EXECUTABLE} ${infile} -o ${outHeaderFile}
            MAIN_DEPENDENCY ${infile})
        set(${outfiles} ${${outfiles}} ${outHeaderFile})
    endforeach()
endmacro()

function(UseQt)
    set(ENV{QTDIR} "${SOFA-EXTERNAL_QT_PATH}")
    set(ENV{CONFIG} "qt;uic")
    find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qt3support qtxml REQUIRED)
    set(QT_QMAKE_EXECUTABLE ${QT_QMAKE_EXECUTABLE} CACHE INTERNAL "QMake executable path")

    include(${QT_USE_FILE})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})

    file(GLOB QT_INCLUDE_SUBDIRS "${QT_INCLUDE_DIR}/Qt*")
    foreach(QT_INCLUDE_SUBDIR ${QT_INCLUDE_SUBDIRS})
        if(IS_DIRECTORY ${QT_INCLUDE_SUBDIR})
            include_directories(${QT_INCLUDE_SUBDIR})
        endif()
    endforeach()

    set(ADDITIONAL_COMPILER_DEFINES ${ADDITIONAL_COMPILER_DEFINES} ${QT_DEFINITIONS} PARENT_SCOPE)
    set(ADDITIONAL_LINKER_DEPENDENCIES ${ADDITIONAL_LINKER_DEPENDENCIES} ${QT_LIBRARIES} PARENT_SCOPE)
endfunction()

# RegisterProjects(<lib0> [lib1 [lib2 ...]] [OPTION <optionName>] [COMPILE_DEFINITIONS <compileDefinition0> [compileDefinition1 [compileDefinition2 ...]] [PATH <path>])
# register a dependency in the dependency tree, used to be retrieved at the end of the project configuration
# to add include directories from dependencies and to enable dependencies / plugins
# libN is a list of library using the same OPTION to be enabled (opengl/glu for instance)
# optionName is the name of the OPTION used to enable / disable the module (for instance SOFA-EXTERNAL_GLEW)
# compiler definitions is the preprocessor macro that has to be globally setted if the project is enabled
# path parameter is the path to the cmake project if any (may be needed to enable the project)
function(RegisterProjects)
    set(dependencies)
    set(optionName "")
    set(noOptionName "")
    set(compilerDefinitions "")
    set(projectPath "")

    set(mode 0)
    foreach(arg ${ARGV})
        if(${arg} STREQUAL "OPTION")
            set(mode 1)
        elseif(${arg} STREQUAL "NO_OPTION")
            set(mode 2)
        elseif(${arg} STREQUAL "COMPILE_DEFINITIONS")
            set(mode 3)
        elseif(${arg} STREQUAL "PATH")
            set(mode 4)
        else()
            if(${mode} EQUAL 0)# libN parameters
                set(dependencies ${dependencies} ${arg})
            elseif(${mode} EQUAL 1) # OPTION parameter
                set(mode 5)
                set(optionName ${arg})
            elseif(${mode} EQUAL 2) # NO_OPTION parameter
                set(mode 5)
                set(noOptionName ${arg})
            elseif(${mode} EQUAL 3) # COMPILE_DEFINITIONS parameter
                set(compilerDefinitions ${compilerDefinitions} ${arg})
            elseif(${mode} EQUAL 4) # PATH parameter
                set(mode 5)
                set(projectPath ${arg})
            elseif(${mode} EQUAL 5) # too many arguments
                message(SEND_ERROR "RegisterProjects(${ARGV}): too many arguments")
                break()
            endif()
        endif()
    endforeach()

    foreach(dependency ${dependencies})
        unset(GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${dependency} CACHE) # if this flag is raised, it means this dependency is up-to-date regarding its dependencies and theirs include directories
        list(FIND GLOBAL_DEPENDENCIES ${dependency} index)
        if(index EQUAL -1)
            set(GLOBAL_DEPENDENCIES ${GLOBAL_DEPENDENCIES} ${dependency} CACHE INTERNAL "Global dependencies" FORCE)
        endif()
        if(NOT optionName STREQUAL "")
            set(GLOBAL_PROJECT_OPTION_${dependency} ${optionName} CACHE INTERNAL "${dependency} options" FORCE)
        endif()
        if(NOT noOptionName STREQUAL "")
            set(GLOBAL_PROJECT_NO_OPTION_${dependency} ${noOptionName} CACHE INTERNAL "${dependency} options to disable if this project is active" FORCE)
        endif()
        if(NOT projectPath STREQUAL "")
            file(TO_CMAKE_PATH "${projectPath}" projectPath)
            get_filename_component(extension "${projectPath}" EXT)
            if(extension STREQUAL ".cmake") # *.cmake subproject are not allowed, we must use a CMakeLists.txt
                message(SEND_ERROR "Including project ${dependency} from a *project_file*.cmake is not supported, you must use a CMakeLists.txt")
            else()
                set(GLOBAL_PROJECT_PATH_${dependency} ${projectPath} CACHE INTERNAL "${dependency} path" FORCE)

                set(includeDirs ${GLOBAL_PROJECT_PATH_${dependency}})
                if(EXISTS "${GLOBAL_PROJECT_PATH_${dependency}}/include")
                    set(includeDirs ${includeDirs} "${GLOBAL_PROJECT_PATH_${dependency}}/include")
                endif()
                set(${dependency}_INCLUDE_DIR ${includeDirs} CACHE INTERNAL "${PROJECT_NAME} include path" FORCE)
            endif()
        endif()
        if(NOT compilerDefinitions STREQUAL "")
            set(GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${dependency} ${compilerDefinitions} CACHE INTERNAL "${dependency} compiler definitions" FORCE)
        endif()
        if(NOT SOFA_PROJECT_FOLDER STREQUAL "")
            #message("== ${dependency} in ${SOFA_PROJECT_FOLDER}")
            set(GLOBAL_PROJECT_OPTION_FOLDER_${dependency} ${SOFA_PROJECT_FOLDER} CACHE INTERNAL "${dependency} solution folder" FORCE)
        endif()
    endforeach()
endfunction()

# AddCompilerDefinitionsFromProject(<dependency>)
# retrieve the compiler defines set when the dependency has been registered (using RegisterProjects and not every compiler defines set when the dependency is being generated) and add it in the current compiler defines
function(AddCompilerDefinitionsFromProject)
    foreach(dependency ${ARGV})
        set(COMPILER_DEFINES ${COMPILER_DEFINES} ${GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${dependency}} PARENT_SCOPE)
    endforeach()
endfunction()

# # EnableProject(<projectName>)
# # enable a project to be generated
# # projectName is the name of a registered project
# function(EnableProject projectName)
#     if(NOT projectName STREQUAL "")
#         #list(FIND GLOBAL_DEPENDENCIES ${projectName} index)
#         #if(NOT index EQUAL -1)
#         message(STATUS " - ${projectName}: Enabled")
#         message(STATUS "")
#         set(GLOBAL_PROJECT_ENABLED_${projectName} 1 CACHE INTERNAL "${projectName} Enabled Status" FORCE)
#         #else()
#         #message(AUTHOR_WARNING "Trying to enable a non registered project: ${projectName}")
#         #endif()
#     else()
#         message(AUTHOR_WARNING "EnableProject error: The project name is empty")
#     endif()
# endfunction()

function(EnableDependencyOption projectName)
    #enable the project option
    if(GLOBAL_PROJECT_OPTION_${projectName})
        if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
            sofa_log_warning("Adding needed project option: ${GLOBAL_PROJECT_OPTION_${projectName}}")
            sofa_force_reconfigure()

            get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_OPTION_${projectName}} PROPERTY HELPSTRING)
            set(${GLOBAL_PROJECT_OPTION_${projectName}} 1 CACHE BOOL "${variableDocumentation}" FORCE)
        endif()
    endif()

    # disable the project no option
    if(GLOBAL_PROJECT_NO_OPTION_${projectName})
        if(NOT ${${GLOBAL_PROJECT_NO_OPTION_${projectName}}})
            get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_NO_OPTION_${projectName}} PROPERTY HELPSTRING)
            set(${GLOBAL_PROJECT_NO_OPTION_${projectName}} 0 CACHE BOOL "${variableDocumentation}" FORCE)

            sofa_log_warning("Disabling option: ${GLOBAL_PROJECT_NO_OPTION_${projectName}}")
            sofa_force_reconfigure()
        endif()
    endif()
endfunction()

# RegisterProjectDependencies(<projectName>)
# register a target and its dependencies
function(RegisterProjectDependencies projectName)
    # dependencies
    set(projectDependencies ${ARGN})
    if(projectDependencies)
        list(REMOVE_DUPLICATES projectDependencies)
        list(REMOVE_ITEM projectDependencies "debug" "optimized" "general") # remove cmake keywords from dependencies
    endif()
    set(GLOBAL_PROJECT_DEPENDENCIES_${projectName} ${projectDependencies} CACHE INTERNAL "${projectName} Dependencies" FORCE)

    # retrieve compile definitions
    get_target_property(compilerDefines ${projectName} COMPILE_DEFINITIONS)
    set(GLOBAL_PROJECT_COMPILER_DEFINITIONS_${projectName} ${compilerDefines} CACHE INTERNAL "${projectName} compile definitions" FORCE)

    # if we manually added an optional project to be generated, we must set its option to ON and its no option to OFF
    EnableDependencyOption(${projectName})

    RegisterProjects(${projectName})

    set(GLOBAL_PROJECT_ENABLED_${projectName} 1 CACHE INTERNAL "${projectName} Enabled Status" FORCE)
endfunction()

# RetrieveDependencies()
function(RetrieveDependencies projectsPath optionPrefix optionDescription definitionPrefix)
    if(ARGV4 STREQUAL "RECURSIVE")
        set(RECURSIVE ON)
    endif()

    file(GLOB dependencyDirs "${projectsPath}/*")
    foreach(dependencyDir ${dependencyDirs})
        if(IS_DIRECTORY ${dependencyDir})
            get_filename_component(dependencyName ${dependencyDir} NAME)
            string(TOUPPER ${dependencyName} dependencyToUpperName)
            set(optionName "${optionPrefix}${dependencyToUpperName}")

            if(NOT dependencyToUpperName STREQUAL "DEPRECATED")
                if(NOT RECURSIVE) # treat the CMakeLists.txt like a Solution
                    file(GLOB solutionPath "${dependencyDir}/CMakeLists.txt")
                    if(solutionPath)
                        string(REPLACE "/CMakeLists.txt" "" solutionFolder ${solutionPath})
                        get_filename_component(solutionName ${solutionFolder} NAME)
                        if(NOT dependencyToUpperName MATCHES ".*_TEST.*")
                            sofa_option("${optionName}" BOOL OFF "${optionDescription} ${dependencyName}")
                        endif()
                        RegisterProjects(${solutionName} OPTION "${optionName}" COMPILE_DEFINITIONS "${definitionPrefix}${dependencyToUpperName}" PATH "${solutionFolder}")
                    endif()
                    unset(solutionPath)
                else() # register every CMakeLists.txt in the folder and in the DIRECT sub-folders (we do not go below the direct sub-folders !)
                    file(GLOB projectPath "${dependencyDir}/CMakeLists.txt")
                    if(projectPath)
                        string(REPLACE "/CMakeLists.txt" "" projectFolder ${projectPath})
                        get_filename_component(projectName ${projectFolder} NAME)
                        sofa_option("${optionName}" BOOL OFF "${optionDescription} ${dependencyName}")
                        RegisterProjects(${projectName} OPTION "${optionName}" COMPILE_DEFINITIONS "${definitionPrefix}${dependencyToUpperName}" PATH "${projectFolder}")

                        # then gather CMakeLists in every direct sub-folders
                        file(GLOB subPathes "${dependencyDir}/*")
                        if(NOT subPathes STREQUAL "")
                            foreach(subPath ${subPathes})
                                if(IS_DIRECTORY ${subPath})
                                    file(GLOB subProject "${subPath}/CMakeLists.txt")
                                    if(NOT subProject STREQUAL "")
                                        string(REPLACE "/CMakeLists.txt" "" subFolder ${subProject})
                                        get_filename_component(subProjectName ${subFolder} NAME)
                                        RegisterProjects(${subProjectName} OPTION "${optionName}" COMPILE_DEFINITIONS "${definitionPrefix}${dependencyToUpperName}" PATH "${subFolder}")
                                    endif()
                                endif()
                            endforeach()
                        endif()
                    endif()
                    unset(projectPath)


                endif()
            endif()
        endif()
    endforeach()
endfunction()

# ComputeDependencies(<projectName>)
# compute project dependencies to enable needed plugins / dependencies and to add theirs include directories
# <projectName> the project to compute
# <forceEnable> if true : this dependency is needed in a project and we need to enable it even if the user disabled it
#if false : this dependency is not needed for now and the user choose to disable, we skip it
# <fromProject>for log purpose only, the name of the project needing the processed dependency
# <offset>for log purpose only, add characters before outputting a line in the log (useful for tree visualization)
function(ComputeDependencies projectName forceEnable fromProject offset)
    string(TOUPPER ${projectName} projectToUpperName)

    unset(isUnitTest)
    if(projectToUpperName MATCHES ".*_TEST.*")
        set(isUnitTest 1)
    endif()

    set(check 1)
    # check if the project is enabled or not
    if(NOT ${forceEnable})
        if(NOT GLOBAL_PROJECT_ENABLED_${projectName})
            if(GLOBAL_PROJECT_OPTION_${projectName})
                if(NOT DEFINED ${GLOBAL_PROJECT_OPTION_${projectName}})
                    if(NOT DEFINED isUnitTest)
                        set(check 0)
                    endif()
                endif()
                if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
                    set(check 0)
                endif()
            else()
                set(check 0)
            endif()
        endif()
    endif()
    # check if the project is a test, if this is the case but SOFA-MISC_TESTS is disabled we do not enable the project
    if(check EQUAL 1 AND NOT SOFA-MISC_TESTS AND isUnitTest)
        set(check 0)
    endif()
    # process the project
    if(check EQUAL 1)
        # process the project if it has not been processed yet
        if(NOT GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${projectName})
            # enable the needed disabled dependency
            EnableDependencyOption(${projectName})

            # also register global compiler definitions - will be added to every projects at the end of the projects configuration
            #if(GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${projectName})
            #set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${projectName}} CACHE INTERNAL "Global Compiler Defines" FORCE)
            #endif()

            # add the current project
            if(GLOBAL_PROJECT_PATH_${projectName}) # TODO: if there is no path try a find_package / find_library
                if(NOT ${GLOBAL_PROJECT_PATH_${projectName}} STREQUAL "")
                    if(SOFA-MISC_CMAKE_VERBOSE)
                        message(STATUS "Adding project '${projectName}' for '${fromProject}' from: ${GLOBAL_PROJECT_PATH_${projectName}}")
                    else()
                        message(STATUS "Adding project '${projectName}'")
                    endif()
                    add_subdirectory("${GLOBAL_PROJECT_PATH_${projectName}}")
                endif()
            endif()

            # mark project as "processed", doing this now avoid dead-lock during recursion in case of circular dependency
            set(GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${projectName} 1 CACHE INTERNAL "${projectName} know all its dependencies status" FORCE)

            # retrieve its dependencies
            set(dependencies ${GLOBAL_PROJECT_DEPENDENCIES_${projectName}})

            # and compute its compiler definitions
            set(dependenciesCompilerDefines)

            #message(STATUS "${offset} + ${projectName}")
            foreach(dependency ${dependencies})
                ComputeDependencies(${dependency} true "${projectName}" "${offset} ")

                set(dependenciesCompilerDefines ${dependenciesCompilerDefines} ${GLOBAL_PROJECT_REGISTERED_COMPILER_DEFINITIONS_${dependency}})
            endforeach()
            #message(STATUS "${offset} - ${projectName}")

            # set the updated compiler definitions of the current project
            set(definitions_list ${dependenciesCompilerDefines} ${GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${projectName}})
            sofa_remove_duplicates(definitions_list)
            set(GLOBAL_PROJECT_REGISTERED_COMPILER_DEFINITIONS_${projectName} ${definitions_list} CACHE INTERNAL "${projectName} compiler definitions from option and dependencies options" FORCE)

            if(TARGET ${projectName})
                unset(compilerDefines)
                list(APPEND compilerDefines ${${projectName}_COMPILER_DEFINES} ${GLOBAL_PROJECT_REGISTERED_COMPILER_DEFINITIONS_${projectName}})
                sofa_remove_duplicates(compilerDefines)
                set_target_properties(${projectName} PROPERTIES COMPILE_DEFINITIONS "${compilerDefines}")
            endif()

            # retrieve include directories from the current project and its dependencies
            list(APPEND includeDirs ${${projectName}_INCLUDE_DIR})
            foreach(dependency ${dependencies})
                if(${dependency}_INCLUDE_DIR)
                    list(APPEND includeDirs ${${dependency}_INCLUDE_DIR})
                    sofa_remove_duplicates(includeDirs)
                endif()
            endforeach()

            # set the updated include directories of the current project
            set(${projectName}_INCLUDE_DIR ${includeDirs} CACHE INTERNAL "${projectName} include path" FORCE)
            if(TARGET ${projectName})
                set_target_properties(${projectName} PROPERTIES INCLUDE_DIRECTORIES "${${projectName}_INCLUDE_DIR}")
            endif()
        endif()
    endif()
endfunction()

# AddCompilerDefinitions(compiler_definition0 [compiler_definition1 [...]])
function(AddCompilerDefinitions)
    set(COMPILER_DEFINES ${COMPILER_DEFINES} ${ARGV} PARENT_SCOPE)
endfunction()

# AddCompilerFlags(compiler_flag0 [compiler_flag1 [...]])
function(AddCompilerFlags)
    set(COMPILER_FLAGS ${COMPILER_FLAGS} ${ARGV} PARENT_SCOPE)
endfunction()

# AddSourceDependencies(source_dependency0 [source_dependency1 [...]])
function(AddSourceDependencies)
    set(SOURCE_DEPENDENCIES ${SOURCE_DEPENDENCIES} ${ARGV} PARENT_SCOPE)
endfunction()

# AddLinkerDependencies(linker_dependency0 [linker_dependency1 [...]])
function(AddLinkerDependencies)
    set(LINKER_DEPENDENCIES ${LINKER_DEPENDENCIES} ${ARGV} PARENT_SCOPE)
endfunction()

# AddLinkerFlags(linker_flag0 [linker_flag1 [...]])
function(AddLinkerFlags)
    set(LINKER_FLAGS ${LINKER_FLAGS} ${ARGV} PARENT_SCOPE)
endfunction()

# Set SOFA_FORCE_RECONFIGURE to signal that CMake must be run again
function(sofa_force_reconfigure)
    set(SOFA_FORCE_RECONFIGURE 1 CACHE INTERNAL "" FORCE)
endfunction()
unset(SOFA_FORCE_RECONFIGURE CACHE) # Reset flag

# Print a warning message and store it. A summary of the warnings is printed at
# the end of the configuration step with sofa_print_list()
function(sofa_log_warning message)
    set(SOFA_WARNING_MESSAGES ${SOFA_WARNING_MESSAGES} "${message}" CACHE INTERNAL "" FORCE)
    message(WARNING "\n${message}\n")
endfunction()
unset(SOFA_WARNING_MESSAGES CACHE) # Clear warning list

# Print an error message and store it. A summary of the errors is printed at
# the end of the configuration step with sofa_print_list()
function(sofa_log_error message)
    set(SOFA_ERROR_MESSAGES ${SOFA_ERROR_MESSAGES} "${message}" CACHE INTERNAL "" FORCE)
    message(SEND_ERROR "\n${message}\n")
endfunction()
unset(SOFA_ERROR_MESSAGES CACHE) # Clear error list

# Print a list of messages with a little bit of formatting
function(sofa_print_list title message_list)
    if(message_list)
        message("> ${title}:")
        foreach(message ${message_list})
            message("  - ${message}")
        endforeach()
    endif()
endfunction()

function(sofa_save_dependencies filename)
    set(text "")

    set(projectNames ${GLOBAL_DEPENDENCIES})
    foreach(projectName ${projectNames})
        if(TARGET ${projectName})
            set(dependencies ${GLOBAL_PROJECT_DEPENDENCIES_${projectName}})
            if (dependencies)
                set(text "${text}> ${projectName} depends on:\n")
                foreach(dependency ${dependencies})
                    set(text "${text}  - ${dependency}\n")
                endforeach()
            else()
                set(text "${text}> ${projectName} has no dependencies\n")
            endif()
        endif()
    endforeach()
    file(WRITE "${SOFA_BUILD_DIR}/${filename}"
"Here are the direct dependencies for every project:

${text}")
    message(STATUS "The list of direct dependencies was saved to: ${filename}")
endfunction()

# Write to a file the list of compilation definitions
function(sofa_save_compiler_definitions filename)
    # Get the definitions for the first TARGET project,
    foreach(project_name ${GLOBAL_DEPENDENCIES})
        if(TARGET ${project_name})
            set(common_definition_list ${GLOBAL_PROJECT_COMPILER_DEFINITIONS_${project_name}})
            break()
        endif()
    endforeach()
    # and find the list of definitions which are common to every project
    foreach(project_name ${GLOBAL_DEPENDENCIES})
        if(TARGET ${project_name})
            sofa_list_intersection(new_list common_definition_list GLOBAL_PROJECT_COMPILER_DEFINITIONS_${project_name})
            set(common_definition_list ${new_list})
        endif()
    endforeach()

    # List, for each project, the definitions which are not in the common list
    set(project_list)
    foreach(project_name ${GLOBAL_DEPENDENCIES})
        if(TARGET ${project_name})
            sofa_list_subtraction(defines GLOBAL_PROJECT_COMPILER_DEFINITIONS_${project_name} common_definition_list)
            set(project_list "${project_list}- ${project_name}: ${defines}\n")
        endif()
    endforeach()

    if("${common_definition_list}" STREQUAL "")
        set(common_definition_list "(none)")
    endif()

    file(WRITE "${SOFA_BUILD_DIR}/${filename}"
        "Every project is compiled with the following definitions:

${common_definition_list}

And here are the project-specific compiler definitions:

${project_list}")
    message(STATUS "The list of compiler definitions was saved to: ${filename}")
endfunction()

function(sofa_print_configuration_report)
    if(SOFA_ERROR_MESSAGES OR SOFA_WARNING_MESSAGES)
        message("")
        message(STATUS "Log summary:")
        sofa_print_list("Errors" "${SOFA_ERROR_MESSAGES}")
        sofa_print_list("Warnings" "${SOFA_WARNING_MESSAGES}")
        message("")
    endif()
    if(NOT SOFA_ERROR_MESSAGES AND SOFA_FORCE_RECONFIGURE)
        message(">>> The configuration has changed, you must configure the project again")
        message("")
    endif()
endfunction()


# Iteratively retrieve all the dependencies of 'project' and store them in 'out_dependency_list'
function(sofa_get_complete_dependencies project out_dependency_list)
    set(current_list)

    set(new_dependencies ${GLOBAL_PROJECT_DEPENDENCIES_${project}})
    while(new_dependencies)
        list(APPEND current_list ${new_dependencies})
        # Get these new_dependencies' own dependencies
        set(dependencies_of_dependencies)
        foreach(name ${new_dependencies})
            list(APPEND dependencies_of_dependencies ${GLOBAL_PROJECT_DEPENDENCIES_${name}})
        endforeach()
        sofa_remove_duplicates(dependencies_of_dependencies)
        # But keep only the new ones
        sofa_list_subtraction(new_dependencies dependencies_of_dependencies current_list)
    endwhile()

    set(${out_dependency_list} ${current_list} PARENT_SCOPE)
endfunction()

function(sofa_save_complete_dependencies filename)
    foreach(projectName ${GLOBAL_DEPENDENCIES})
        if(TARGET ${projectName})
            sofa_get_complete_dependencies(${projectName} dependencies)
            set(targets)
            set(others)
            foreach(dependency ${dependencies})
                if(TARGET ${dependency})
                    list(APPEND targets "${dependency}")
                else()
                    list(APPEND others "${dependency}")
                endif()
            endforeach()
            if(NOT targets)
                set(targets "(none)")
            endif()
            if(NOT others)
                set(others "(none)")
            endif()
            set(text "${text}> ${projectName}:\n- Targets:\n${targets}\n- Others:\n${others}\n\n")
        endif()
    endforeach()
    file(WRITE "${SOFA_BUILD_DIR}/${filename}" "For debugging purposes, here is the list of ALL dependencies for each project.\n\n${text}")
    message(STATUS "The list of complete dependencies was saved to: ${filename}")
endfunction()

function(sofa_add_plugin plugin_name)
    add_library("${plugin_name}" SHARED ${ARGN})
    set_target_properties("${plugin_name}" PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${SOFA_BIN_PLUGINS_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_PLUGINS_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_PLUGINS_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${SOFA_BIN_PLUGINS_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${SOFA_BIN_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${SOFA_BIN_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${SOFA_BIN_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${SOFA_BIN_PLUGINS_DIR}")
endfunction()
