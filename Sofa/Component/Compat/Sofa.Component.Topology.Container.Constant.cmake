set(SOFABASETOPOLOGY_SRC src/SofaBaseTopology)
set(SOFAGENERALTOPOLOGY_SRC src/SofaGeneralTopology)

list(APPEND HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/MeshTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/CubeTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/SphereQuadTopology.h
)
