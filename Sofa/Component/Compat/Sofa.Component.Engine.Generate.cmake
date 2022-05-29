set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)
set(SOFAMISCEXTRA_SRC src/SofaMiscExtra)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.h
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.inl
    ${SOFAGENERALENGINE_SRC}/JoinPoints.h
    ${SOFAGENERALENGINE_SRC}/JoinPoints.inl
    ${SOFAGENERALENGINE_SRC}/MergePoints.h
    ${SOFAGENERALENGINE_SRC}/MergePoints.inl
    ${SOFAGENERALENGINE_SRC}/MergeSets.h
    ${SOFAGENERALENGINE_SRC}/MergeSets.inl
    ${SOFAGENERALENGINE_SRC}/MergeVectors.h
    ${SOFAGENERALENGINE_SRC}/MergeVectors.inl
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.h
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.inl
    ${SOFAGENERALENGINE_SRC}/NormEngine.h
    ${SOFAGENERALENGINE_SRC}/NormEngine.inl
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.h
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeEdgesAndGenerateQuads.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeEdgesAndGenerateQuads.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeQuadsAndGenerateHexas.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeQuadsAndGenerateHexas.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeSurface.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeSurface.inl
    ${SOFAGENERALENGINE_SRC}/GenerateCylinder.h
    ${SOFAGENERALENGINE_SRC}/GenerateCylinder.inl
    ${SOFAGENERALENGINE_SRC}/GenerateGrid.h
    ${SOFAGENERALENGINE_SRC}/GenerateGrid.inl
    ${SOFAGENERALENGINE_SRC}/GenerateSphere.h
    ${SOFAGENERALENGINE_SRC}/GenerateSphere.inl
    ${SOFAGENERALENGINE_SRC}/MergeMeshes.h
    ${SOFAGENERALENGINE_SRC}/MergeMeshes.inl
    ${SOFAGENERALENGINE_SRC}/MeshBarycentricMapperEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshBarycentricMapperEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshClosingEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshClosingEngine.inl
    ${SOFAGENERALENGINE_SRC}/Spiral.h
    ${SOFAGENERALENGINE_SRC}/Spiral.inl
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.h
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.inl
    ${SOFAMISCEXTRA_SRC}/MeshTetraStuffing.h
)
