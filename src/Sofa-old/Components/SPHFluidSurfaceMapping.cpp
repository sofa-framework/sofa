#include "SPHFluidSurfaceMapping.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"
#include "Sofa-old/Core/MappedModel.h"
#include "Sofa-old/Core/MechanicalModel.h"
#include "Sofa-old/Core/MechanicalMapping.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

SOFA_DECL_CLASS(SPHFluidSurfaceMapping)

template<class In, class Out>
void create(SPHFluidSurfaceMapping<In,Out>*& obj, ObjectDescription* arg)
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
//Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > > > SPHFluidSurfaceMapping3d3dClass("SPHFluidSurfaceMapping", true);
//Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > > > SPHFluidSurfaceMapping3f3fClass("SPHFluidSurfaceMapping", true);
//Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > > > SPHFluidSurfaceMapping3d3fClass("SPHFluidSurfaceMapping", true);
//Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > > > SPHFluidSurfaceMapping3f3dClass("SPHFluidSurfaceMapping", true);

// Mech -> Mapped
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > > SPHFluidSurfaceMapping3dM3dClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > > SPHFluidSurfaceMapping3fM3fClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > > SPHFluidSurfaceMapping3dM3fClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > > SPHFluidSurfaceMapping3fM3dClass("SPHFluidSurfaceMapping", true);

// Mech -> ExtMapped
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > SPHFluidSurfaceMapping3dME3dClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > SPHFluidSurfaceMapping3fME3fClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > SPHFluidSurfaceMapping3dME3fClass("SPHFluidSurfaceMapping", true);
Creator< ObjectFactory, SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > SPHFluidSurfaceMapping3fME3dClass("SPHFluidSurfaceMapping", true);


// Mech -> Mech
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > >;
//template class SPHFluidSurfaceMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > >;

// Mech -> Mapped
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> >;

// Mech -> ExtMapped
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;

} // namespace Components

} // namespace Sofa
