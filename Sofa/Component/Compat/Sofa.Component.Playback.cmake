set(SOFAGENERALLOADER_SRC src/SofaGeneralLoader)
set(SOFAEXPORTER_SRC src/SofaExporter)
set(SOFAVALIDATION_SRC src/SofaValidation)

list(APPEND HEADER_FILES
    ${SOFAGENERALLOADER_SRC}/ReadState.h
    ${SOFAGENERALLOADER_SRC}/ReadState.inl
    ${SOFAGENERALLOADER_SRC}/ReadTopology.h
    ${SOFAGENERALLOADER_SRC}/ReadTopology.inl
    ${SOFAGENERALLOADER_SRC}/InputEventReader.h
    ${SOFAEXPORTER_SRC}/WriteState.h
    ${SOFAEXPORTER_SRC}/WriteState.inl
    ${SOFAEXPORTER_SRC}/WriteTopology.h
    ${SOFAEXPORTER_SRC}/WriteTopology.inl
    ${SOFAVALIDATION_SRC}/CompareState.h
    ${SOFAVALIDATION_SRC}/CompareTopology.h
)
