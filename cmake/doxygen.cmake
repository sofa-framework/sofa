
# In this file, if the option SOFA-MISC_DOXYGEN in enabled, we create
# targets to generate doxygen documentation for each module and
# plugin, namely:
#
# - the 'doc-Foo' target generates the doxygen for the Foo module or
#   plugin (e.g, doc-SofaEngine, or doc-SofaPython).  Also, doc-Foo
#   depends on doc-Bar if Foo depends on the Bar module/plugin.
#
# - the 'doc-SOFA' target generates the documentation for framework/,
#   and inserts in its main page a list of links to the others
#   documentations.
#
# - the 'doc' target depends on all the doc-* targets.


# Furthermore, if the SOFA-MISC_DOXYGEN_COMPONENT_LIST option is
# enabled, building 'doc' will compile Sofa and generate an extra page
# in the main documentation that list all the components available in
# modules and links to their documentation page, so that one can
# easily find the doxygen of a component.


# Details:
#
# Doxygen has mechanisms that allow to link documentations together.
# Here is the outline: when generating the documentation for projectA,
# you can export the list of documented entities in a tag file. Then,
# when generating the documentation for another project which uses
# projectA, you can feed this tag file to doxygen. As a result,
# symbols from projectA present in your documentation are linked to
# their documentation in projectA.
#
# (see http://www.stack.nl/~dimitri/doxygen/manual/external.html)
#
# Here, we use this 'tag file' mechanism of doxygen and the dependency
# tree of the projects to generate the documentation of the different
# projects in a correct order, and link them together.


if (SOFA-MISC_DOXYGEN)
    find_package(Doxygen REQUIRED)

    file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/doc")
    file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/misc/doxygen-tagfiles")
    file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/misc/doxyfiles")


    # First thing, we build the list of the projects for which we will
    # generate a doxygen documentation.
    set(SOFA_DOCUMENTABLE_PROJECTS)
    set(SOFA_DOC_TARGETS)
    foreach(project ${GLOBAL_DEPENDENCIES})
        if((${GLOBAL_PROJECT_PATH_${project}} MATCHES ".*/plugins/${project}") OR
                (${GLOBAL_PROJECT_PATH_${project}} MATCHES ".*/modules/${project}"))
            # Ignore SofaComponent* meta-modules
            if(NOT ${project} MATCHES "SofaComponent.*")
                list(APPEND SOFA_DOCUMENTABLE_PROJECTS ${project})
                list(APPEND SOFA_DOC_TARGETS doc-${project})
            endif()
        endif()
    endforeach()


    # Function: create a 'doc-${name}' target to build the documentation for ${input}
    # name: a name for the documentation (also used as the title)
    # input: source files or directories
    # dependencies: the names of the documentations that must be generated before this one
    function(add_doc_target name input dependencies)

        # Build the list of targets we depend on, and the list of corresponding tag files
        set(dependencies_targets)
        set(tag_files)
        foreach(dependency ${dependencies})
            list(APPEND dependencies_targets "doc-${dependency}")
            set(tag_files "${tag_files} misc/doxygen-tagfiles/${dependency}=../${dependency}")
        endforeach()

        # Generate a Doxyfile based on Doxyfile.in, with configure_file()
        # (if VAR is set, the string @VAR@ in Doxyfile.in will be replaced with ${VAR})
        set(TAGFILES "${tag_files}")
        set(INPUT "${input}")
        set(NAME "${name}")
        # If this is not the main page, include the custom header with a link to the main page
        if(NOT ${name} STREQUAL "SOFA")
            set(HTML_HEADER_FILE "${SOFA_CMAKE_DIR}/doxygen/header.html")
        endif()
        set(HTML_STYLESHEET_FILE "${SOFA_CMAKE_DIR}/doxygen/stylesheet.css")
        set(HTML_OUTPUT "${name}")
        set(GENERATE_TAGFILE "misc/doxygen-tagfiles/${name}")
        configure_file("${SOFA_CMAKE_DIR}/doxygen/Doxyfile.in" "${SOFA_BUILD_DIR}/misc/doxyfiles/Doxyfile-${name}")

        # Create the 'doc-${name}' target, which calls doxygen with the Doxyfile we generated
        add_custom_target("doc-${name}"
            COMMAND ${DOXYGEN_EXECUTABLE} "${SOFA_BUILD_DIR}/misc/doxyfiles/Doxyfile-${name}"
            DEPENDS ${dependencies_targets})
        # Put the target in a folder 'Documentation' (for IDEs)
        set_target_properties("doc-${name}" PROPERTIES FOLDER "Documentation")
    endfunction()

    # Create documentation targets for all the projects we choose to document
    foreach(project ${SOFA_DOCUMENTABLE_PROJECTS})
        sofa_get_complete_dependencies(${project} project_dependencies)
        sofa_list_intersection(documentable_dependencies project_dependencies SOFA_DOCUMENTABLE_PROJECTS)
        set(input "${GLOBAL_PROJECT_PATH_${project}}")

        add_doc_target("${project}" "${input}" "${documentable_dependencies};SOFA")
    endforeach()


    # Use configure_file() to generate the source for the main page of
    # the documentation, which lists all the other documentations.

    set(DOCUMENTATION_TEXT "")
    macro(doc_append txt)
        set(DOCUMENTATION_TEXT "${DOCUMENTATION_TEXT}${txt}\n")
    endmacro()

    # Macro: create a list of links to other documentations,
    # based on the path of the projects
    macro(doc_append_list category pattern)
        # Filter projects: keep only those whose path match 'pattern'
        set(filtered_list)
        foreach(project ${SOFA_DOCUMENTABLE_PROJECTS})
            if(${GLOBAL_PROJECT_PATH_${project}} MATCHES "${pattern}")
                list(APPEND filtered_list ${project})
            endif()
        endforeach()
        if(filtered_list)
            doc_append("  <ul>")
            foreach(project ${filtered_list})
                if(${GLOBAL_PROJECT_PATH_${project}} MATCHES "${pattern}")
                    doc_append("    <li><a href=\"../${project}/index.html\"><b>${project}</b></a></li>")
                endif()
            endforeach()
            doc_append("  </ul>")
        endif()
    endmacro()

    set(DOCUMENTATION_TEXT "")
    doc_append_list("Plugins" ".*/plugins/.*")
    configure_file("cmake/doxygen/sofa_plugins.h.in" "${SOFA_BUILD_DIR}/misc/sofa_plugins.h")

    set(DOCUMENTATION_TEXT "")
    doc_append_list("Modules" ".*/modules/.*")
    configure_file("cmake/doxygen/sofa_modules.h.in" "${SOFA_BUILD_DIR}/misc/sofa_modules.h")

    if(SOFA-MISC_DOXYGEN_COMPONENT_LIST)
        add_subdirectory(cmake/doxygen)
    endif()

    # Create the 'doc-SOFA' target for the documentation of framework/
    if(SOFA-MISC_DOXYGEN_COMPONENT_LIST)
        add_custom_target("component_list"
            COMMAND bin/generateComponentList > misc/sofa_modules_component_list.h
            DEPENDS generateComponentList)
    else()
        add_custom_target("component_list"
            COMMAND cp ${SOFA_CMAKE_DIR}/doxygen/empty_sofa_modules_component_list.h ${CMAKE_BINARY_DIR}/misc/sofa_modules_component_list.h)
    endif()
    add_doc_target("SOFA" "${SOFA_FRAMEWORK_DIR} ${SOFA_MODULES_DIR}/sofa ${CMAKE_BINARY_DIR}/misc/sofa_modules.h ${CMAKE_BINARY_DIR}/misc/sofa_plugins.h ${CMAKE_BINARY_DIR}/misc/sofa_modules_component_list.h" "")
    add_dependencies("doc-SOFA" "component_list")
    set_target_properties("doc-SOFA" PROPERTIES FOLDER "Documentation") # IDE Folder

    # Create the 'doc' target, to build every documentation
    add_custom_target("doc" DEPENDS ${SOFA_DOC_TARGETS} "doc-SOFA")
    set_target_properties("doc" PROPERTIES FOLDER "Documentation") # IDE Folder

    # Create a convenient shortcut to the main page
    if(NOT WIN32)
        execute_process(COMMAND ln -sf SOFA/index.html doc/index.html)
    endif()

endif()
