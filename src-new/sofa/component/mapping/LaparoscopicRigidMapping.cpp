#include <sofa/component/mapping/LaparoscopicRigidMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(LaparoscopicRigidMapping)

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

template<class BaseMapping>
void create(LaparoscopicRigidMapping<BaseMapping>*& obj, simulation::tree::xml::ObjectDescription* arg)
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

Creator<simulation::tree::xml::ObjectFactory, LaparoscopicRigidMapping< MechanicalMapping< MechanicalState<LaparoscopicRigidTypes>, MechanicalState<RigidTypes> > > > LaparoscopicRigidMappingClass("LaparoscopicRigidMapping", true);
Creator<simulation::tree::xml::ObjectFactory, LaparoscopicRigidMapping< Mapping< MechanicalState<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > > > LaparoscopicRigidMechanicalMappingClass("LaparoscopicRigidMapping", true);

template class LaparoscopicRigidMapping< MechanicalMapping<MechanicalState<LaparoscopicRigidTypes>, MechanicalState<RigidTypes> > >;
template class LaparoscopicRigidMapping< Mapping<MechanicalState<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

