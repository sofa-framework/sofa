#include <sofa/component/mapping/SPHFluidSurfaceMapping.inl>
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

SOFA_DECL_CLASS(SPHFluidSurfaceMapping)

template<class In, class Out>
void create(SPHFluidSurfaceMapping<In,Out>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWith2Objects< SPHFluidSurfaceMapping<In,Out>, In, Out>(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("radius"))
            obj->setRadius(atof(arg->getAttribute("radius")));
        if (arg->getAttribute("step"))
            obj->setStep(atof(arg->getAttribute("step")));
        if (arg->getAttribute("isoValue"))
            obj->setIsoValue(atof(arg->getAttribute("isoValue")));
    }
}

// Mech -> Mech
//Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > > SPHFluidSurfaceMapping3d3dClass("SPHFluidSurfaceMapping", true);
//Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > > SPHFluidSurfaceMapping3f3fClass("SPHFluidSurfaceMapping", true);
//Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > > SPHFluidSurfaceMapping3d3fClass("SPHFluidSurfaceMapping", true);
//Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > > SPHFluidSurfaceMapping3f3dClass("SPHFluidSurfaceMapping", true);

// Mech -> Mapped
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > > SPHFluidSurfaceMapping3dM3dClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > > SPHFluidSurfaceMapping3fM3fClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > > SPHFluidSurfaceMapping3dM3fClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > > SPHFluidSurfaceMapping3fM3dClass("SPHFluidSurfaceMapping", true);

// Mech -> ExtMapped
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > SPHFluidSurfaceMapping3dME3dClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > SPHFluidSurfaceMapping3fME3fClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > SPHFluidSurfaceMapping3dME3fClass("SPHFluidSurfaceMapping", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > SPHFluidSurfaceMapping3fME3dClass("SPHFluidSurfaceMapping", true);


// Mech -> Mech
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> >;

// Mech -> ExtMapped
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;

} // namespace mapping

} // namespace component

} // namespace sofa

