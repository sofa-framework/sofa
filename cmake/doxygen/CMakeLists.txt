include_directories("${SOFA_FRAMEWORK_DIR}" "${SOFA_MODULES_DIR}")
add_executable(generateComponentList generateComponentList.cpp)
target_link_libraries(generateComponentList SofaComponentMain)
