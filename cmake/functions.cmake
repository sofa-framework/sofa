cmake_minimum_required(VERSION 2.8)

# group files
macro(GroupFiles fileGroup topGroup)	
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

# generate .h / .cpp from Qt3 .ui for Qt4
macro(QT4_UIC3_WRAP_UI outfiles )
	QT4_EXTRACT_OPTIONS(ui_files ui_options ${ARGN})

	foreach(it ${ui_files})
		get_filename_component(outfile ${it} NAME_WE)
		get_filename_component(infile ${it} ABSOLUTE)
		set(outHeaderFile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}.h")
		set(outSourceFile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}.cpp")
		add_custom_command(	OUTPUT ${outHeaderFile} ${outSourceFile}
							COMMAND ${QT_UIC3_EXECUTABLE} ${ui_options} ${infile} -o ${outHeaderFile}
							COMMAND ${QT_UIC3_EXECUTABLE} ${ui_options} "-impl" ${outHeaderFile} ${infile} -o ${outSourceFile}
							MAIN_DEPENDENCY ${infile})
		
		QT4_WRAP_CPP(outMocFile ${outHeaderFile})
		set(${outfiles} ${${outfiles}} ${outHeaderFile} ${outSourceFile} ${outMocFile})
	endforeach()
endmacro()

# generate .h / .cpp from Qt3 .ui for Qt3
macro(QT3_UIC3_WRAP_UI outfiles )
	QT3_EXTRACT_OPTIONS(ui_files ui_options ${ARGN})

	foreach(it ${ui_files})
		get_filename_component(outfile ${it} NAME_WE)
		get_filename_component(infile ${it} ABSOLUTE)
		set(outHeaderFile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}.h")
		set(outSourceFile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}.cpp")
		add_custom_command(	OUTPUT ${outHeaderFile} ${outSourceFile}
							COMMAND ${QT_UIC3_EXECUTABLE} ${ui_options} ${infile} -o ${outHeaderFile}
							COMMAND ${QT_UIC3_EXECUTABLE} ${ui_options} "-impl" ${outHeaderFile} ${infile} -o ${outSourceFile}
							MAIN_DEPENDENCY ${infile})
							
		QT3_WRAP_CPP(outMocFile ${outHeaderFile})
		set(${outfiles} ${${outfiles}} ${outHeaderFile} ${outSourceFile} ${outMocFile})
	endforeach()
endmacro()

function(UseQt)
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

# RegisterDependencies(<lib0> [lib1 [lib2 ...]] [OPTION <optionName> [<path>]])
# register a dependency in the dependency tree, used to be retrieved at the end of the project configuration
# to add include directories from dependencies and to enable dependencies / plugins
# libN is a list of library using the same OPTION to be enabled (opengl/glu for instance)
# optionName is the name of the OPTION used to enable / disable the module (for instance EXTERNAL_HAVE_GLEW)
# path parameter is the path to the cmake project if any (may be needed to enable the project)
function(RegisterDependencies)
	set(dependencies)
	set(mode 0)
	set(optionName "")
	set(projectPath "")
	
	foreach(arg ${ARGV})
		if(${mode} EQUAL 0)	# libN parameters
			if(${arg} STREQUAL "OPTION")
				set(mode 1)
			else()
				list(FIND GLOBAL_DEPENDENCIES ${arg} index)
				if(index EQUAL -1)
					set(dependencies ${dependencies} ${arg})
				endif()
			endif()
		elseif(${mode} EQUAL 1) # OPTION arguments
			set(mode 2)
			set(optionName ${arg})
		elseif(${mode} EQUAL 2) # PATH parameter
			set(mode 3)
			set(projectPath ${arg})
		elseif(${mode} EQUAL 3) # too many arguments
			message(SEND_ERROR "RegisterDependencies(${ARGV}) : too many arguments")
			break()
		endif()
	endforeach()

	foreach(dependency ${dependencies})
		unset(GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${dependency} CACHE) # if this flag is raised, it means this dependency is up-to-date regarding its dependencies and theirs include directories
		set(GLOBAL_DEPENDENCIES ${GLOBAL_DEPENDENCIES} ${dependency} CACHE INTERNAL "Global checked dependencies" FORCE)
		if(NOT optionName STREQUAL "")
			set(GLOBAL_PROJECT_OPTION_${dependency} ${optionName} CACHE INTERNAL "${dependency} options" FORCE)
			if(NOT projectPath STREQUAL "")
				set(GLOBAL_PROJECT_PATH_${dependency} ${projectPath} CACHE INTERNAL "${dependency} path" FORCE)
			endif()
		endif()
	endforeach()
endfunction()

# RegisterProjectDependencies(<projectName> [lib0 [lib1 [lib2 ...]]])
# register a target and its dependencies
function(RegisterProjectDependencies projectName)
	set(projectDependencies ${ARGN})
	list(LENGTH projectDependencies projectDependenciesNum)
	if(NOT projectDependenciesNum EQUAL 0)
		list(REMOVE_DUPLICATES projectDependencies)
		list(REMOVE_ITEM projectDependencies "debug" "optimized" "general") # remove cmake keywords from dependencies
	endif()
	set(GLOBAL_PROJECT_DEPENDENCIES_${projectName} ${projectDependencies} CACHE INTERNAL "${projectName} Dependencies" FORCE)
	
	RegisterDependencies(${projectName})
endfunction()

# ComputeDependencies(<projectName>)
# compute project dependencies to enable needed plugins / dependencies and to add theirs include directories
# <projectName> the project to compute
# <forceEnable> if true : this dependency is needed in a project and we need to enable it even if the user disabled it
#				if false : this dependency is not needed for now and the user choose to disable, we skip it
# <offset>		for log purpose only, add characters before outputting a line in the log (useful for tree visualization)
function(ComputeDependencies projectName forceEnable offset)
	set(check 1)
	# check if the project is enabled or not
	if((NOT ${forceEnable}) AND GLOBAL_PROJECT_OPTION_${projectName})
		if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
			set(check 0)
		endif()
	endif()
	# process the project
	if(check EQUAL 1)
		# process the project if it has not been processed yet
		if(NOT GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${projectName})
			# enable the needed disabled dependency
			if(GLOBAL_PROJECT_OPTION_${projectName})
				if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
					get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_OPTION_${projectName}} PROPERTY HELPSTRING)
					set(${GLOBAL_PROJECT_OPTION_${projectName}} 1 CACHE BOOL "${variableDocumentation}" FORCE)
					# add the current project
					if(GLOBAL_PROJECT_PATH_${projectName})
						if(NOT ${GLOBAL_PROJECT_PATH_${projectName}} STREQUAL "")
							message(STATUS "")
							message(STATUS "${offset}  Adding needed option : ${GLOBAL_PROJECT_OPTION_${projectName}}")
							message(STATUS "")
							
							add_subdirectory("${GLOBAL_PROJECT_PATH_${projectName}}")
						endif()
					endif()
					# TODO: if there is no path try a find_package / find_library
				endif()
			endif()
			
			# mark project as "processed", doing this now avoid dead-lock during recursion in case of circular dependency
			set(GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${projectName} 1 CACHE INTERNAL "${projectName} know all its dependencies status" FORCE)
		
			# retrieve its dependencies and intersect with the known ones (which may be currently disabled)
			listIntersection(dependencies GLOBAL_PROJECT_DEPENDENCIES_${projectName} GLOBAL_DEPENDENCIES)
			
			message(STATUS "${offset} + ${projectName}")
			foreach(dependency ${dependencies})
				ComputeDependencies(${dependency} true "${offset} ")
			endforeach()
			message(STATUS "${offset} - ${projectName}")
			
			# retrieve include directories
			list(APPEND includeDirs ${${projectName}_INCLUDE_DIR})
			foreach(dependency ${dependencies})
				if(TARGET ${dependency})
					if(${dependency}_INCLUDE_DIR)
						list(APPEND includeDirs ${${dependency}_INCLUDE_DIR})
						list(REMOVE_DUPLICATES includeDirs)
					endif()
				endif()
			endforeach()
			set(${projectName}_INCLUDE_DIR ${includeDirs} CACHE INTERNAL "${projectName} include path" FORCE)
			if(TARGET ${projectName})
				set_property(TARGET ${projectName} PROPERTY INCLUDE_DIRECTORIES ${${projectName}_INCLUDE_DIR})
			endif()
		endif()
	endif()
endfunction()

# create a cache file
function(CreateCacheFile)
	if(GENERATED_FROM_MAIN_SOLUTION)
		set(cacheFilename "${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt")
		file(GLOB cacheFile "${cacheFilename}")
		if(cacheFile STREQUAL "")
			file(WRITE "${cacheFilename}" "SOFA_CMAKE_DIR:INTERNAL=${SOFA_CMAKE_DIR}")
		endif()
	endif()
endfunction()

# low-level function computing an intersection between two lists
macro(listIntersection outList inList0 inList1)
	set(${outList})
	set(tmpList)
	
	foreach(inElem1 ${${inList1}})
		foreach(inElem0 ${${inList0}})
			if(inElem0 STREQUAL ${inElem1})
				set(tmpList ${tmpList} ${inElem0})
			endif()
		endforeach()
	endforeach()
	
	set(${outList} ${tmpList})
endmacro()

# low-level function computing a subtraction between two lists
macro(listSubtraction outList inList0 inList1)
	set(${outList})
	set(tmpList)
	
	foreach(inElem0 ${${inList0}})
		set(add 1)
		foreach(inElem1 ${${inList1}})
			if(inElem0 STREQUAL ${inElem1})
				set(add 0)
				break()
			endif()
		endforeach()
		if(add EQUAL 1)
			set(tmpList ${tmpList} ${inElem0})
		endif()
	endforeach()
	
	set(${outList} ${tmpList})
endmacro()