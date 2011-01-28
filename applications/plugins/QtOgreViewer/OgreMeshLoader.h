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

};

}
}
}


#endif // SOFA_COMPONENT_LOADER_OGREMESHLOADER_H
