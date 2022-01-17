set(SOFATOPOLOGYMAPPING_SRC src/SofaTopologyMapping)

list(APPEND HEADER_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/CenterPointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Edge2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2TetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/IdentityTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Quad2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedHexaTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SubsetTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Tetra2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Triangle2EdgeTopologicalMapping.h
)
