set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.h
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.inl
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.h
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.inl
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.h
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.inl
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.h
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.inl
    ${SOFAGENERALENGINE_SRC}/JoinPoints.h
    ${SOFAGENERALENGINE_SRC}/JoinPoints.inl
    ${SOFAGENERALENGINE_SRC}/MapIndices.h
    ${SOFAGENERALENGINE_SRC}/MapIndices.inl
    ${SOFAGENERALENGINE_SRC}/MergePoints.h
    ${SOFAGENERALENGINE_SRC}/MergePoints.inl
    ${SOFAGENERALENGINE_SRC}/MergeSets.h
    ${SOFAGENERALENGINE_SRC}/MergeSets.inl
    ${SOFAGENERALENGINE_SRC}/MergeVectors.h
    ${SOFAGENERALENGINE_SRC}/MergeVectors.inl
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.h
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ROIValueMapper.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.inl
)
