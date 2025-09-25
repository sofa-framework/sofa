cmake_minimum_required(VERSION 3.22)

# Remove this when https://github.com/sofa-framework/SofaGLFW/pull/235 will be merged
#sofa_add_external(directory SofaGLFW GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaGLFW.git)
sofa_add_external(directory SofaGLFW GIT_REF fix_cmake GIT_REPOSITORY https://www.github.com/olivier-roussel/SofaGLFW.git)
sofa_add_external(plugin Sofa.Qt GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/Sofa.Qt.git)
