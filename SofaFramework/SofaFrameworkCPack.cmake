###########################################
# Sofa :: Framework CPack configuration   #
###########################################

if(UNIX)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libglew-dev freeglut3-dev libpng-dev")
elseif(WIN32)
    set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP")
endif()

set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../readme.txt")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copyright.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${SOFAFRAMEWORK_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${SOFAFRAMEWORK_MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "${SOFAFRAMEWORK_BUILD_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SofaFramework libraries")
set(CPACK_PACKAGE_MAINTAINER "INRIA") #required
set(CPACK_PACKAGE_CONTACT "sofa-users@lists.gforge.inria.fr")
 
set(CPACK_COMPONENTS_ALL SofaFramework_sources SofaFramework_libraries )
set(CPACK_COMPONENT_SOFAFRAMEWORK_LIBRARIES_DISPLAY_NAME "libs" )
set(CPACK_COMPONENT_SOFAFRAMEWORK_SOURCES_DISPLAY_NAME "c++ sources" )
set(CPACK_COMPONENT_SOFAFRAMEWORK_LIBRARIES_GROUP "SofaFramework" )
set(CPACK_COMPONENT_SOFAFRAMEWORK_SOURCES_GROUP   "SofaFramework" )
set(CPACK_COMPONENT_GROUP_SOFAFRAMEWORK_DESCRIPTION "The SofaFramework package") 



include(CPack)