cmake_minimum_required(VERSION 2.8)
project(MultiThreading)

set(MULTITHREADING_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(MULTITHREADING_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(MULTITHREADING_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
	#src/Observer.h
    src/initMultiThreading.h
	src/DataExchange.h
	src/DataExchange.inl)

set(SOURCE_FILES
	#src/Observer.cpp
    src/initMultiThreading.cpp
	src/DataExchange.cpp)

find_package(SofaMisc REQUIRED)
find_package(Boost COMPONENTS thread)

if(Boost_FOUND)
	list(APPEND HEADER_FILES "src/TaskSchedulerBoost.h")
	list(APPEND SOURCE_FILES "src/TaskSchedulerBoost.cpp")

	list(APPEND HEADER_FILES "src/Tasks.h")
	list(APPEND HEADER_FILES "src/Tasks.inl")
	list(APPEND SOURCE_FILES "src/Tasks.cpp")


    list(APPEND HEADER_FILES "src/AnimationLoopParallelScheduler.h")
	list(APPEND SOURCE_FILES "src/AnimationLoopParallelScheduler.cpp")

    list(APPEND HEADER_FILES "src/AnimationLoopTasks.h")
	list(APPEND SOURCE_FILES "src/AnimationLoopTasks.cpp")

    #list(APPEND HEADER_FILES "src/ParallelForTask.h")
    #list(APPEND SOURCE_FILES "src/ParallelForTask.cpp")

	list(APPEND HEADER_FILES "src/BeamLinearMapping_mt.h")
	list(APPEND HEADER_FILES "src/BeamLinearMapping_mt.inl")
	list(APPEND SOURCE_FILES "src/BeamLinearMapping_mt.cpp")
	list(APPEND HEADER_FILES "src/BeamLinearMapping_tasks.inl")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping)

## uncomment this to use the intel threading building block library
#set(SOFA_HAVE_TBB true)

# # Useful ?
# find_package(SofaCUDA)
# if(SofaCUDA_FOUND)
#     target_link_libraries(${PROJECT_NAME} SofaCUDA)
# endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT MultiThreading_libraries
    EXPORT MultiThreadingTargets
    RUNTIME DESTINATION "${MULTITHREADING_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${MULTITHREADING_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${MULTITHREADING_INSTALL_LIB_DIR}")
