set(SOFABASETOPOLOGY_SRC src/SofaBaseTopology)
set(SOFAGENERALTOPOLOGY_SRC src/SofaGeneralTopology)
set(SOFANONUNIFORMFEM_SRC src/SofaNonUniformFem)

list(APPEND HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/polygon_cube_intersection.h
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/vec.h
    ${SOFABASETOPOLOGY_SRC}/GridTopology.h
    ${SOFABASETOPOLOGY_SRC}/RegularGridTopology.h
    ${SOFABASETOPOLOGY_SRC}/SparseGridTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/CylinderGridTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/SphereGridTopology.h
    ${SOFANONUNIFORMFEM_SRC}/SparseGridMultipleTopology.h
    ${SOFANONUNIFORMFEM_SRC}/SparseGridRamificationTopology.h
)
