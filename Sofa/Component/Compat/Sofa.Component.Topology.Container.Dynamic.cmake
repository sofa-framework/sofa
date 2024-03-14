set(SOFABASETOPOLOGY_SRC src/SofaBaseTopology)
set(SOFANONUNIFORMFEM_SRC src/SofaNonUniformFem)

list(APPEND HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/fwd.h
    ${SOFABASETOPOLOGY_SRC}/CommonAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/NumericalIntegrationDescriptor.h
    ${SOFABASETOPOLOGY_SRC}/NumericalIntegrationDescriptor.inl
    ${SOFABASETOPOLOGY_SRC}/PointSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/PointSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyModifier.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridGeometryAlgorithms.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridGeometryAlgorithms.inl
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyAlgorithms.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyContainer.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyModifier.h
    ${SOFANONUNIFORMFEM_SRC}/MultilevelHexahedronSetTopologyContainer.h
)
