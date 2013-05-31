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
	QT4_EXTRACT_OPTIONS(moc_files moc_options ${ARGN})

	set(defines)
	foreach(it ${GLOBAL_COMPILER_DEFINES})
		list(APPEND defines "-D${it}")
	endforeach()

	foreach(it ${moc_files})
	get_filename_component(it ${it} ABSOLUTE)
	QT4_MAKE_OUTPUT_FILE(${it} moc_ cpp outfile)
	QT4_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags}" "${defines}" "${moc_options}")
	set(${outfiles} ${${outfiles}} ${outfile})
	endforeach()
endmacro()

# generate mocced headers from Qt3 moccable headers
macro(SOFA_QT3_WRAP_CPP outfiles )
	# get include dirs
	QT3_GET_MOC_FLAGS(moc_flags)
	QT3_EXTRACT_OPTIONS(moc_files moc_options ${ARGN})

	set(defines)
	foreach(it ${GLOBAL_COMPILER_DEFINES})
		list(APPEND defines "-D${it}")
	endforeach()

	foreach(it ${moc_files})
	get_filename_component(it ${it} ABSOLUTE)
	QT3_MAKE_OUTPUT_FILE(${it} moc_ cpp outfile)
	QT3_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags}" "${defines}" "${moc_options}")
	set(${outfiles} ${${outfiles}} ${outfile})
	endforeach()
endmacro()

# generate .h / .cpp from Qt3 .ui for Qt4
macro(SOFA_QT4_WRAP_UI outfiles )
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
		
		SOFA_QT4_WRAP_CPP(outMocFile ${outHeaderFile})
		set(${outfiles} ${${outfiles}} ${outHeaderFile} ${outSourceFile} ${outMocFile})
	endforeach()
endmacro()

# generate .h / .cpp from Qt3 .ui for Qt3
macro(SOFA_QT3_WRAP_UI outfiles )
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
							
		SOFA_QT3_WRAP_CPP(outMocFile ${outHeaderFile})
		set(${outfiles} ${${outfiles}} ${outHeaderFile} ${outSourceFile} ${outMocFile})
	endforeach()
endmacro()

function(UseQt)
	set(ENV{QTDIR} "${SOFA-EXTERNAL_QT_PATH}")
	if(SOFA-EXTERNAL_PREFER_QT4)
		set(ENV{CONFIG} "qt;uic;uic3")
		find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qt3support qtxml REQUIRED)
	else()
		set(ENV{CONFIG} "qt")
		find_package(Qt3 COMPONENTS qtopengl REQUIRED)
	endif()
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

# RegisterDependencies(<lib0> [lib1 [lib2 ...]] [OPTION <optionName>] [COMPILE_DEFINITIONS <compileDefinition0> [compileDefinition1 [compileDefinition2 ...]] [PATH <path>])
# register a dependency in the dependency tree, used to be retrieved at the end of the project configuration
# to add include directories from dependencies and to enable dependencies / plugins
# libN is a list of library using the same OPTION to be enabled (opengl/glu for instance)
# optionName is the name of the OPTION used to enable / disable the module (for instance SOFA-EXTERNAL_HAVE_GLEW)
# compiler definitions is the preprocessor macro that has to be globally setted if the project is enabled
# path parameter is the path to the cmake project if any (may be needed to enable the project)
function(RegisterDependencies)
	set(dependencies)
	set(optionName)
	set(noOptionName)
	set(compilerDefinitions)
	set(projectPath)
	
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
			if(${mode} EQUAL 0)	# libN parameters
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
				message(SEND_ERROR "RegisterDependencies(${ARGV}) : too many arguments")
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

# EnableProject(<projectName>)
# enable a project to be generated
# projectName is the name of a registered project
function(EnableProject projectName)
	if(NOT projectName STREQUAL "")
		#list(FIND GLOBAL_DEPENDENCIES ${projectName} index)
		#if(NOT index EQUAL -1)
			message(STATUS " - ${projectName} : Enabled")
			set(GLOBAL_PROJECT_ENABLED_${projectName} 1 CACHE INTERNAL "${projectName} Enabled Status" FORCE)
		#else()
		#	message(AUTHOR_WARNING 	"Trying to enable a non registered project : ${projectName}")
		#endif()
	else()
		message(AUTHOR_WARNING "EnableProject error : The project name is empty")
	endif()
endfunction()

# RegisterProjectDependencies(<projectName>)
# register a target and its dependencies
function(RegisterProjectDependencies projectName)
	# dependencies
	set(projectDependencies ${ARGN})
	list(LENGTH projectDependencies projectDependenciesNum)
	if(NOT projectDependenciesNum EQUAL 0)
		list(REMOVE_DUPLICATES projectDependencies)
		list(REMOVE_ITEM projectDependencies "debug" "optimized" "general") # remove cmake keywords from dependencies
	endif()
	set(GLOBAL_PROJECT_DEPENDENCIES_${projectName} ${projectDependencies} CACHE INTERNAL "${projectName} Dependencies" FORCE)
	
	# retrieve compile definitions
	get_target_property(compilerDefines ${projectName} COMPILE_DEFINITIONS)
	set(GLOBAL_PROJECT_COMPILER_DEFINITIONS_${projectName} ${compilerDefines} CACHE INTERNAL "${projectName} compile definitions" FORCE)
		
	# if we manually added an optional project to be generated, we must set its option to ON
	if(GLOBAL_PROJECT_OPTION_${projectName})
		if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
			message(WARNING " - Adding needed project option : ${GLOBAL_PROJECT_OPTION_${projectName}}")
			get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_OPTION_${projectName}} PROPERTY HELPSTRING)
			set(${GLOBAL_PROJECT_OPTION_${projectName}} 1 CACHE BOOL "${variableDocumentation}" FORCE)	
		endif()
	endif()
	
	# if we manually added an optional project to be generated, we must set its no option to OFF
	if(GLOBAL_PROJECT_NO_OPTION_${projectName})
		if(NOT ${${GLOBAL_PROJECT_NO_OPTION_${projectName}}})
			get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_NO_OPTION_${projectName}} PROPERTY HELPSTRING)
			set(${GLOBAL_PROJECT_NO_OPTION_${projectName}} 0 CACHE BOOL "${variableDocumentation}" FORCE)	
		endif()
	endif()
	
	RegisterDependencies(${projectName})
	
	set(GLOBAL_PROJECT_ENABLED_${projectName} 1 CACHE INTERNAL "${projectName} Enabled Status" FORCE)
endfunction()

# ComputeDependencies(<projectName>)
# compute project dependencies to enable needed plugins / dependencies and to add theirs include directories
# <projectName> the project to compute
# <forceEnable> if true : this dependency is needed in a project and we need to enable it even if the user disabled it
#				if false : this dependency is not needed for now and the user choose to disable, we skip it
# <fromProject>	for log purpose only, the name of the project needing the processed dependency
# <offset>		for log purpose only, add characters before outputting a line in the log (useful for tree visualization)
function(ComputeDependencies projectName forceEnable fromProject offset)
	set(check 1)
	# check if the project is enabled or not
	if(NOT ${forceEnable})
		if(NOT GLOBAL_PROJECT_ENABLED_${projectName})
			if(GLOBAL_PROJECT_OPTION_${projectName})
				if(NOT DEFINED ${GLOBAL_PROJECT_OPTION_${projectName}})
					message(FATAL_ERROR "The option : '${GLOBAL_PROJECT_OPTION_${projectName}}' has been used to register a project but does not exist")
				endif()
				if(NOT ${${GLOBAL_PROJECT_OPTION_${projectName}}})
					set(check 0)
				endif()
			else()
				set(check 0)
			endif()
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
					message(WARNING " - Adding needed option : ${GLOBAL_PROJECT_OPTION_${projectName}}")
				endif()
			endif()
			
			# disable the no option
			if(GLOBAL_PROJECT_NO_OPTION_${projectName})
				if(NOT ${${GLOBAL_PROJECT_NO_OPTION_${projectName}}})
					get_property(variableDocumentation CACHE ${GLOBAL_PROJECT_NO_OPTION_${projectName}} PROPERTY HELPSTRING)
					set(${GLOBAL_PROJECT_NO_OPTION_${projectName}} 0 CACHE BOOL "${variableDocumentation}" FORCE)
					message(WARNING " - Disabling option : ${GLOBAL_PROJECT_NO_OPTION_${projectName}}")
				endif()
			endif()
			
			# also register global compiler definitions - will be added to every projects at the end of the projects configuration
			if(GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${projectName})
				set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${GLOBAL_PROJECT_OPTION_COMPILER_DEFINITIONS_${projectName}} CACHE INTERNAL "Global Compiler Defines" FORCE)
			endif()
			
			# add the current project
			if(GLOBAL_PROJECT_PATH_${projectName}) # TODO: if there is no path try a find_package / find_library
				if(NOT ${GLOBAL_PROJECT_PATH_${projectName}} STREQUAL "")
					message(STATUS " - Adding dependency '${projectName}' for '${fromProject}' from : ${GLOBAL_PROJECT_PATH_${projectName}}")
					add_subdirectory("${GLOBAL_PROJECT_PATH_${projectName}}")
				endif()
			endif()
			
			# mark project as "processed", doing this now avoid dead-lock during recursion in case of circular dependency
			set(GLOBAL_PROJECT_DEPENDENCIES_COMPLETE_${projectName} 1 CACHE INTERNAL "${projectName} know all its dependencies status" FORCE)
		
			# retrieve its dependencies
			set(dependencies ${GLOBAL_PROJECT_DEPENDENCIES_${projectName}})
			
			#message(STATUS "${offset} + ${projectName}")
			foreach(dependency ${dependencies})
				ComputeDependencies(${dependency} true "${projectName}" "${offset} ")
			endforeach()
			#message(STATUS "${offset} - ${projectName}")
			
			# retrieve include directories from the current project and its dependencies
			list(APPEND includeDirs ${${projectName}_INCLUDE_DIR})
			#list(REMOVE_DUPLICATES includeDirs)
			foreach(dependency ${dependencies})
				if(TARGET ${dependency})
					if(${dependency}_INCLUDE_DIR)
						list(APPEND includeDirs ${${dependency}_INCLUDE_DIR})
						list(REMOVE_DUPLICATES includeDirs)
					endif()
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

# ApplyGlobalCompilerDefinitions(<projectName>)
# set global compiler definitions to a specific project
# <projectName> the project to compute
function(ApplyGlobalCompilerDefinitions projectName)
	# process the project
	if(TARGET ${projectName})
		set(compilerDefines ${GLOBAL_COMPILER_DEFINES})
		list(APPEND compilerDefines ${GLOBAL_PROJECT_COMPILER_DEFINITIONS_${projectName}})
		list(REMOVE_DUPLICATES compilerDefines)
		set_target_properties(${projectName} PROPERTIES COMPILE_DEFINITIONS "${compilerDefines}")
	endif()
endfunction()

# AddCompilerDefinitions(compiler_definition0 [compiler_definition1 [...]])
function(AddCompilerDefinitions)
	set(COMPILER_DEFINES ${ARGV} PARENT_SCOPE)
endfunction()

# AddCompilerFlags(compiler_flag0 [compiler_flag1 [...]])
function(AddCompilerFlags)
	set(COMPILER_FLAGS ${ARGV} PARENT_SCOPE)
endfunction()

# AddLinkerDependencies(linker_dependency0 [linker_dependency1 [...]])
function(AddLinkerDependencies)
	set(LINKER_DEPENDENCIES ${ARGV} PARENT_SCOPE)
endfunction()

# AddLinkerFlags(linker_flag0 [linker_flag1 [...]])
function(AddLinkerFlags)
	set(LINKER_FLAGS ${ARGV} PARENT_SCOPE)
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
