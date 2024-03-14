set(SOFALOADER_SRC src/SofaLoader)
set(SOFAGENERALLOADER_SRC src/SofaGeneralLoader)
set(SOFAEXPORTER_SRC src/SofaExporter)

list(APPEND HEADER_FILES
    ${SOFALOADER_SRC}/BaseVTKReader.h
    ${SOFALOADER_SRC}/BaseVTKReader.inl
    ${SOFALOADER_SRC}/MeshOBJLoader.h
    ${SOFALOADER_SRC}/MeshVTKLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshGmshLoader.h
    ${SOFAGENERALLOADER_SRC}/GIDMeshLoader.h
    ${SOFAGENERALLOADER_SRC}/GridMeshCreator.h
    ${SOFAGENERALLOADER_SRC}/MeshOffLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshSTLLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshTrianLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshXspLoader.h
    ${SOFAGENERALLOADER_SRC}/OffSequenceLoader.h
    ${SOFAGENERALLOADER_SRC}/SphereLoader.h
    ${SOFAGENERALLOADER_SRC}/StringMeshCreator.h
    ${SOFAGENERALLOADER_SRC}/VoxelGridLoader.h
    ${SOFAEXPORTER_SRC}/BlenderExporter.h
    ${SOFAEXPORTER_SRC}/BlenderExporter.inl
    ${SOFAEXPORTER_SRC}/MeshExporter.h
    ${SOFAEXPORTER_SRC}/STLExporter.h
    ${SOFAEXPORTER_SRC}/VisualModelOBJExporter.h
    ${SOFAEXPORTER_SRC}/VTKExporter.h
)
