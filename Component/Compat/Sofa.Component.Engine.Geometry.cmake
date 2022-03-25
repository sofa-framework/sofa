set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)
set(SOFAMISCENGINE_SRC src/SofaMiscEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.h
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.inl
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.h
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.inl
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.h
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.inl
    ${SOFAGENERALENGINE_SRC}/NormEngine.h
    ${SOFAGENERALENGINE_SRC}/NormEngine.inl
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.h
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.inl
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.h
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.inl
    ${SOFAMISCENGINE_SRC}/Distances.h
    ${SOFAMISCENGINE_SRC}/Distances.inl
)
