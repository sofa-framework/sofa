#include "LaparoscopicRigidMapping.inl"
#include "Common/Vec3Types.h"
#include "Common/RigidTypes.h"
#include "Common/LaparoscopicRigidTypes.h"
#include "Common/ObjectFactory.h"
#include "Sofa/Core/MappedModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Core/MechanicalMapping.inl"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(LaparoscopicRigidMapping)

using namespace Common;
using namespace Core;

template<class BaseMapping>
void create(LaparoscopicRigidMapping<BaseMapping>*& obj, ObjectDescription* arg)
{
    XML::createWith2Objects< LaparoscopicRigidMapping<BaseMapping>, typename LaparoscopicRigidMapping<BaseMapping>::In, typename LaparoscopicRigidMapping<BaseMapping>::Out>(obj, arg);
    if (obj!=NULL)
    {
        {
            Vector3 val;
            const char* str = arg->getAttribute("pivot","0 0 0");
            sscanf(str, "%lf %lf %lf", &(val[0]),&(val[1]),&(val[2]));
            obj->setPivot(val);
        }
        {
            Quat val;
            const char* str = arg->getAttribute("rotation","0 0 0 1");
            sscanf(str, "%lf %lf %lf %lf", &(val[0]),&(val[1]),&(val[2]),&(val[3]));
            obj->setRotation(val);
        }
    }
}

Creator< ObjectFactory, LaparoscopicRigidMapping< MechanicalMapping< MechanicalModel<LaparoscopicRigidTypes>, MechanicalModel<RigidTypes> > > > LaparoscopicRigidMappingClass("LaparoscopicRigidMapping", true);
Creator< ObjectFactory, LaparoscopicRigidMapping< Mapping< MechanicalModel<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > > > LaparoscopicRigidMechanicalMappingClass("LaparoscopicRigidMapping", true);

template class LaparoscopicRigidMapping< MechanicalMapping<MechanicalModel<LaparoscopicRigidTypes>, MechanicalModel<RigidTypes> > >;
template class LaparoscopicRigidMapping< Mapping<MechanicalModel<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > >;

} // namespace Components

} // namespace Sofa
