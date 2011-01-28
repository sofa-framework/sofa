#ifndef SOFA_COMPONENT_LOADER_OGREMESHLOADER_H
#define SOFA_COMPONENT_LOADER_OGREMESHLOADER_H

#include <sofa/core/loader/MeshLoader.h>
#include <Ogre.h>

namespace sofa
{
namespace component
{
namespace loader
{

class OgreMeshLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(OgreMeshLoader,sofa::core::loader::MeshLoader);

    OgreMeshLoader();

    bool load();

protected:
    bool readMesh(Ogre::Mesh* mesh);
    void readMeshVertices(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector3 >& vertices);
    void readMeshNormals(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector3 >& normal);
    void readMeshTexCoords(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector2>& coord);
    //void readMeshIndices(Ogre::VertexData* vertexData, helper::vector< helper::fixed_array <unsigned int,3> >& indices);

    Data< helper::vector<sofa::defaulttype::Vector2> > texCoords;

};

}
}
}


#endif // SOFA_COMPONENT_LOADER_OGREMESHLOADER_H
