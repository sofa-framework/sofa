get_name_from_source_dir() #This macro will define two variables {inner-project-name} {inner-project-name-upper} from the name of the directory containing this file

set(${inner-project-name-upper}_GIT_REPOSITORY "https://forge.icube.unistra.fr/sofa/${inner-project-name}.git" CACHE STRING "Repository address" )
set(${inner-project-name-upper}_GIT_TAG "${ARG_GIT_REF}" CACHE STRING "Branch or commit SHA to checkout" )