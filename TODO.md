TEMPLATE FOR THE CHANGE.

################################ COMPONENTS HERE ARE THE NG-SET ####################################


############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES 
        
        )
    list(APPEND SOURCE_FILES 
        
        )
    message(STATUS "SofaGeneralEngine: build all maintained sofa components.")
else()
    message(STATUS "SofaGeneralEngine: build with only the minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    #list(APPEND HEADER_FILES MeshG.h)
    #list(APPEND SOURCE_FILES MeshGenerationFromImage.cpp)
    message(STATUS "SofaGeneralEngine: build with deprecated components.")
else()
    message(STATUS "SofaGeneralEngine: build without deprecated components.")
endif()

