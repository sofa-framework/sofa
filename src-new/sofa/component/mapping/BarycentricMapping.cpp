#include <sofa/component/mapping/BarycentricMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(BarycentricMapping)

template<class BaseMapping>
void create(BarycentricMapping<BaseMapping>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWith2Objects< BarycentricMapping<BaseMapping>, typename BarycentricMapping<BaseMapping>::In, typename BarycentricMapping<BaseMapping>::Out>(obj, arg);
}


// Mech -> Mech
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > > BarycentricMapping3d3dClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > > BarycentricMapping3f3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > > BarycentricMapping3d3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > > BarycentricMapping3f3dClass("BarycentricMapping", true);

// Mech -> Mapped
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > > > BarycentricMapping3dM3dClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > > > BarycentricMapping3fM3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > > > BarycentricMapping3dM3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > > > BarycentricMapping3fM3dClass("BarycentricMapping", true);

// Mech -> ExtMapped
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > > BarycentricMapping3dME3dClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > > BarycentricMapping3fME3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > > BarycentricMapping3dME3fClass("BarycentricMapping", true);
Creator<simulation::tree::xml::ObjectFactory, BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > > BarycentricMapping3fME3dClass("BarycentricMapping", true);

// Mech -> Mech
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

