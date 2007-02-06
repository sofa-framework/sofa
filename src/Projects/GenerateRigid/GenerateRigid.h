#ifndef GENERATERIGID_H
#define GENERATERIGID_H

#include "Sofa-old/Components/Common/Mesh.h"
#include "Sofa-old/Components/Common/RigidTypes.h"
#include "Sofa-old/Components/Common/Vec.h"

namespace Projects
{

bool GenerateRigid(Sofa::Components::Common::RigidMass& mass, Sofa::Components::Common::Vec3d& center, Sofa::Components::Common::Mesh* mesh);

}

#endif
