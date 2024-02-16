cmake_minimum_required(VERSION 3.12)

find_package(Sofa.Config REQUIRED)

sofa_add_subdirectory(directory SofaHighOrder SofaHighOrder EXTERNAL GIT_REF master)
sofa_add_subdirectory(directory SofaPython3 SofaPython3 EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CGALPlugin CGALPlugin EXTERNAL GIT_REF master)     # Depends on image
sofa_add_subdirectory(plugin Registration Registration EXTERNAL GIT_REF master) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_subdirectory(plugin InvertibleFVM InvertibleFVM EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin PluginExample PluginExample EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ManifoldTopologies ManifoldTopologies EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin BeamAdapter BeamAdapter EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin STLIB STLIB EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SoftRobots SoftRobots EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CollisionAlgorithm CollisionAlgorithm EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ConstraintGeometry ConstraintGeometry EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ShapeMatchingPlugin ShapeMatchingPlugin EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CSparseSolvers CSparseSolvers EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin PSL PSL EXTERNAL GIT_REF master)
