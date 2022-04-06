set(SOFABOUNDARYCONDITION_SRC src/SofaBoundaryCondition)

list(APPEND HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.inl
)
