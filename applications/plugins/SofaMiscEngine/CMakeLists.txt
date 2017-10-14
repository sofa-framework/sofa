cmake_minimum_required(VERSION 3.1)
project(SofaMiscEngine)

set(SOFAMISCENGINE_VERSION 1.0)

################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES config.h initMiscEngine.h)
set(SOURCE_FILES initMiscEngine.cpp)
set(EXTRA_FILES README.md)

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES)
    list(APPEND SOURCE_FILES)
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND HEADER_FILES
        DisplacementTransformEngine.h
        DisplacementTransformEngine.inl
        Distances.h
        Distances.inl
        ProjectiveTransformEngine.h
        ProjectiveTransformEngine.inl
        )
    list(APPEND SOURCE_FILES
        DisplacementTransformEngine.cpp
        Distances.cpp
        ProjectiveTransformEngine.cpp
        )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    ## Add here deprecated component. 
    list(APPEND HEADER_FILES
        DisplacementMatrixEngine.h
        DisplacementMatrixEngine.inl
        )
    list(APPEND SOURCE_FILES
        DisplacementMatrixEngine.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaNonUniformFem)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_ENGINE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

# Install rules for the library and headers; This generate a CMake package configurations files
# so that other plugins can load this one. 
#sofa_create_package(SofaEulerianFluid ${SOFATEST_VERSION} ${PROJECT_NAME} SofaEulerianFluid)



