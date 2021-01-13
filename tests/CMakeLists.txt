cmake_minimum_required(VERSION 3.12)

# Here are the tests that must be compiled AFTER all modules
# usually because they need to run a simulation (functional tests)

set(SOFA_EXT_MODULES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/modules)
set(SOFA_EXT_MODULES_BINARY_DIR ${CMAKE_BINARY_DIR})

set(SOFA_KERNEL_MODULES_SOURCE_DIR ${SOFA_KERNEL_SOURCE_DIR}/modules)
set(SOFA_KERNEL_MODULES_BINARY_DIR ${CMAKE_BINARY_DIR}/SofaKernel)

# SofaFramework tests
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaHelper/SofaHelper_simutest ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaFramework/SofaHelper/SofaHelper_simutest)
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaDefaultType/SofaDefaultType_simutest ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaFramework/SofaDefaultType/SofaDefaultType_simutest)
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaCore/SofaCore_simutest ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaFramework/SofaCore/SofaCore_simutest)
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaSimulationCore/SofaSimulationCore_simutest ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaFramework/SofaSimulationCore/SofaSimulationCore_simutest)

# SofaSimulation tests
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaSimulationGraph/SofaSimulationGraph_test ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaSimulation/SofaSimulationGraph/SofaSimulationGraph_test)
#add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaSimulationCommon/SofaSimulationCommon_test ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaSimulation/SofaSimulationCommon/SofaSimulationCommon_test)

# SofaBase tests
add_subdirectory(${SOFA_KERNEL_MODULES_SOURCE_DIR}/SofaBaseUtils/SofaBaseUtils_test ${SOFA_KERNEL_MODULES_BINARY_DIR}/SofaBase/SofaBaseVisual/SofaBaseUtils_test)
