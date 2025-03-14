get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE "\." "_"  fixed_name ${ProjectId})
string(TOUPPER ${fixed_name} fixed_name)

set(${fixed_name}_GIT_REPOSITORY "https://www.github.com/sofa-framework/${ProjectId}.git" CACHE STRING "Repository address" )
set(${fixed_name}_GIT_TAG "${ARG_GIT_REF}" CACHE STRING "Branch or commit SHA to checkout" )