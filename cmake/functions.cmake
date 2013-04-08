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
