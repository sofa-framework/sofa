#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMulti2Mapping_CPP

#include <sofa/component/mapping/CenterOfMassMulti2Mapping.inl>
#include <sofa/core/Multi2Mapping.inl>
#include <sofa/core/behavior/MechanicalMulti2Mapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(CenterOfMassMulti2Mapping)

using namespace sofa::defaulttype;
using namespace core::behavior;


// Register in the Factory
int CenterOfMassMulti2MappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< CenterOfMassMulti2Mapping< MechanicalMulti2Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CenterOfMassMulti2Mapping< MechanicalMulti2Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping< MechanicalMulti2Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping< MechanicalMulti2Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
