cmake_minimum_required(VERSION 3.22)

#Add SofaPython3 first to enable all bindings in plugins
sofa_add_external(directory SofaPython3 GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaPython3.git)

#Add guis second to enable GUI dependent codes in plugins
add_subdirectory(guis)

#Add plugins before project so project can depend on plugins
add_subdirectory(plugins)
add_subdirectory(projects)
