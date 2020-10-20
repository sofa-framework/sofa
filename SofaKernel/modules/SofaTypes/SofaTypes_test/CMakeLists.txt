cmake_minimum_required(VERSION 3.12)

project(SofaTypestest)

set(SOURCE_FILES
    
)


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaHelper SofaTypes)

add_test(NAME SofaTypes_test COMMAND SofaTypes_test)
