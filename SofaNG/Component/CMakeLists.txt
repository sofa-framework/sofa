cmake_minimum_required(VERSION 3.12)


macro(sofang_add_component_subdirectory DirectoryArg)
	set(DirectoryName ${DirectoryArg})
	set(ExtraMacroArgs ${ARGN})
	# Get the length of the list
  	list(LENGTH ExtraMacroArgs NumExtraMacroArgs)
  	if(NumExtraMacroArgs GREATER 0)
  		set(DirectoryName ${ARGV1})
	else()
		set(DirectoryName Sofa.Component.${DirectoryName})
  	endif()

	string(TOUPPER ${DirectoryName} UpperDirectoryName)
	string(REPLACE "." "_" UpperDirectoryName ${UpperDirectoryName})

	option(SOFANG_ENABLE_${UpperDirectoryName} "Build ${DirectoryName}." ON)
	if(SOFANG_ENABLE_${UpperDirectoryName})
		add_subdirectory(${DirectoryArg})
	endif()
endmacro()

macro(sofang_check_dependencies)
	if(${ARGC} EQUAL 0)
		message(FATAL_ERROR "sofang_check_dependencies macro needs at least one argument")
	endif()
	set(list_deps "${ARGN}")
	foreach(dep IN LISTS list_deps)
		if(NOT TARGET ${dep})
			message(FATAL_ERROR "This target needs ${dep}")
		endif()
	endforeach()
endmacro()

# just enable Compat if already not activated
# (if using the cmakelists .in SofaNG/.)
if(NOT TARGET Sofa.Component.Compat)
	add_subdirectory(Compat)
endif()

# sofang_add_component_subdirectory(Utils)
sofang_add_component_subdirectory(ODESolver)
# sofang_add_component_subdirectory(LinearSolver)
# sofang_add_component_subdirectory(Topology)
# sofang_add_component_subdirectory(PhysicalModel)
# sofang_add_component_subdirectory(Constraint)
# sofang_add_component_subdirectory(AnimationLoop) # Needs Constraint
# sofang_add_component_subdirectory(Mapping)
# sofang_add_component_subdirectory(Container)
# sofang_add_component_subdirectory(Collision)
# sofang_add_component_subdirectory(Engine)
# sofang_add_component_subdirectory(Visual)
# sofang_add_component_subdirectory(IO)

# sofang_add_component_subdirectory(GUI/Core)
# sofang_add_component_subdirectory(Controller)
