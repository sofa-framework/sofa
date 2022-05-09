set(SOFABASEMECHANICS_SRC src/SofaBaseMechanics)
set(SOFARIGID_SRC src/SofaRigid)
set(SOFAGENERALRIGID_SRC src/SofaGeneralRigid)
set(SOFATOPOLOGICALMAPPING_SRC src/SofaTopologyMapping)
set(SOFAMISCMAPPING_SRC src/SofaMiscMapping)

list(APPEND HEADER_FILES
    ${SOFABASEMECHANICS_SRC}/BarycentricMapping.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMapping.inl
    ${SOFABASEMECHANICS_SRC}/IdentityMapping.h
    ${SOFABASEMECHANICS_SRC}/IdentityMapping.inl
    ${SOFABASEMECHANICS_SRC}/SubsetMapping.h
    ${SOFABASEMECHANICS_SRC}/SubsetMapping.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapper.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapper.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/TopologyBarycentricMapper.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/TopologyBarycentricMapper.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperMeshTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperMeshTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperRegularGridTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperRegularGridTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperSparseGridTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperSparseGridTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTopologyContainer.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTopologyContainer.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperEdgeSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTriangleSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperQuadSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperQuadSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl
    ${SOFAGENERALRIGID_SRC}/LineSetSkinningMapping.h
    ${SOFAGENERALRIGID_SRC}/LineSetSkinningMapping.inl
    ${SOFAGENERALRIGID_SRC}/SkinningMapping.h
    ${SOFAGENERALRIGID_SRC}/SkinningMapping.inl
    ${SOFATOPOLOGICALMAPPING_SRC}/Mesh2PointMechanicalMapping.h
    ${SOFATOPOLOGICALMAPPING_SRC}/Mesh2PointMechanicalMapping.inl
    ${SOFATOPOLOGICALMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFATOPOLOGICALMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFAMISCMAPPING_SRC}/BarycentricMappingRigid.h
    ${SOFAMISCMAPPING_SRC}/BarycentricMappingRigid.inl
    ${SOFAMISCMAPPING_SRC}/BeamLinearMapping.h
    ${SOFAMISCMAPPING_SRC}/BeamLinearMapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMappingOperation.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMulti2Mapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMulti2Mapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/DeformableOnRigidFrameMapping.h
    ${SOFAMISCMAPPING_SRC}/DeformableOnRigidFrameMapping.inl
    ${SOFAMISCMAPPING_SRC}/IdentityMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/IdentityMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/SubsetMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/SubsetMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/TubularMapping.h
    ${SOFAMISCMAPPING_SRC}/TubularMapping.inl
    ${SOFAMISCMAPPING_SRC}/VoidMapping.h
)