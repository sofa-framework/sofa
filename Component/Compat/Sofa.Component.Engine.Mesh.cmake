set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/DilateEngine.h
    ${SOFAGENERALENGINE_SRC}/DilateEngine.inl
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
    ${SOFAGENERALENGINE_SRC}/MeshSampler.h
    ${SOFAGENERALENGINE_SRC}/MeshSampler.inl
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.inl
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.h
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.inl
    ${SOFAGENERALENGINE_SRC}/Spiral.h
    ${SOFAGENERALENGINE_SRC}/Spiral.inl
)