cmake_minimum_required(VERSION 3.22)

sofa_add_subdirectory(application SceneChecking SceneChecking ON)
sofa_add_subdirectory(application Modeler Modeler OFF)

sofa_add_subdirectory(application SofaPhysicsAPI SofaPhysicsAPI)

sofa_add_external(directory SofaGLFW GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaGLFW.git)
sofa_add_external(plugin Sofa.Qt GIT_REF master OFF GIT_REPOSITORY https://www.github.com/sofa-framework/Sofa.Qt.git)
sofa_add_subdirectory(application runSofa runSofa ON)

sofa_add_external(directory Regression GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/Regression.git)
sofa_add_subdirectory(application sofaProjectExample sofaProjectExample)

