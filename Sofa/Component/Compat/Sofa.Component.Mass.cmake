set(SOFABASEMECHANICS_SRC src/SofaBaseMechanics)
set(SOFAMISCFORCEFIELD_SRC src/SofaMiscForceField)

list(APPEND HEADER_FILES    
    ${SOFABASEMECHANICS_SRC}/AddMToMatrixFunctor.h
    ${SOFABASEMECHANICS_SRC}/DiagonalMass.h
    ${SOFABASEMECHANICS_SRC}/DiagonalMass.inl
    ${SOFABASEMECHANICS_SRC}/MassType.h
    ${SOFABASEMECHANICS_SRC}/UniformMass.h
    ${SOFABASEMECHANICS_SRC}/UniformMass.inl
    ${SOFAMISCFORCEFIELD_SRC}/MeshMatrixMass.h
    ${SOFAMISCFORCEFIELD_SRC}/MeshMatrixMass.inl
)
