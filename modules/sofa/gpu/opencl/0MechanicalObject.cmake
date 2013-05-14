set(OBJECT_DIR "OpenCLMechanicalObject" )

list(APPEND HEADER_FILES "OpenCLMechanicalObject.h")
list(APPEND HEADER_FILES "OpenCLMechanicalObject.inl")

list(APPEND SOURCE_FILES "OpenCLMechanicalObject.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLMechanicalObject.cl")
