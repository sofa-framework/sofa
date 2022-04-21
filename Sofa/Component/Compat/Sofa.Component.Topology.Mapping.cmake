set(SOFATOPOLOGYMAPPING_SRC src/SofaTopologyMapping)

list(APPEND HEADER_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/CenterPointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Edge2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2TetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/IdentityTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Quad2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SubsetTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Tetra2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Triangle2EdgeTopologicalMapping.h
)
