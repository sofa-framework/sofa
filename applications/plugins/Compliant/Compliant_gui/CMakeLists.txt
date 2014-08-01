include(${SOFA_CMAKE_DIR}/preProject.cmake)

if (SOFA-LIB_GUI_QT)

set(HEADER_FILES

        QCompliantMouseOperations.h

        )

UseQt()
SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

# these header files do not need MOCcing
list(APPEND HEADER_FILES

        initCompliant_gui.h
        CompliantAttachPerformer.h
        CompliantAttachPerformer.inl

        )
	
	
set(SOURCE_FILES

        initCompliant_gui.cpp
        CompliantAttachPerformer.cpp
        QCompliantMouseOperations.cpp

	)



set(MOC_FILES_GROUP "Generated")


if(WIN32)
	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
else()
	include_directories("/usr/include/suitesparse/")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${MOC_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPLIANT_GUI")
AddLinkerDependencies(SofaGuiMain Compliant)

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
