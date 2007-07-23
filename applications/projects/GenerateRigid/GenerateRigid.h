#ifndef GENERATERIGID_H
#define GENERATERIGID_H

#include <sofa/helper/io/Mesh.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace projects
{

bool GenerateRigid(sofa::defaulttype::Rigid3Mass& mass, sofa::defaulttype::Vec3d& center, sofa::helper::io::Mesh* mesh);

}

#endif
