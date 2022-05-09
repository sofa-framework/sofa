set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)
set(SOFAMISCENGINE_SRC src/SofaMiscEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/AverageCoord.h
    ${SOFAGENERALENGINE_SRC}/AverageCoord.inl
    ${SOFAGENERALENGINE_SRC}/SumEngine.h
    ${SOFAGENERALENGINE_SRC}/SumEngine.inl
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.h
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.inl
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.h
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.inl
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.h
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.inl
    ${SOFAMISCENGINE_SRC}/Distances.h
    ${SOFAMISCENGINE_SRC}/Distances.inl
)
