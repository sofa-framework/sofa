#ifndef GENERATERIGID_H
#define GENERATERIGID_H

#include "Sofa/Components/Common/Mesh.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "Sofa/Components/Common/Vec.h"

bool GenerateRigid(Sofa::Components::Common::RigidMass& mass, Sofa::Components::Common::Vec3d& center, Sofa::Components::Common::Mesh* mesh);

#endif
