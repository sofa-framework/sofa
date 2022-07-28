set(SOFABOUNDARYCONDITION_SRC src/SofaBoundaryCondition)
set(SOFAGENERALOBJECTINTERACTION_SRC src/SofaGeneralObjectInteraction)

list(APPEND HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.inl
)
