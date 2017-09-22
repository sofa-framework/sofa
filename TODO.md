TEMPLATE FOR THE CHANGE.

message(STATUS "SofaBoundaryCondition:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################


############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES 
        
        )
    list(APPEND SOURCE_FILES 
        
        )
    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()

