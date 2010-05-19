#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP

#include <sofa/component/mapping/CenterOfMassMultiMapping.inl>
#include <sofa/core/MultiMapping.inl>
#include <sofa/core/behavior/MechanicalMultiMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(CenterOfMassMultiMapping)

using namespace sofa::defaulttype;
using namespace core::behavior;


// Register in the Factory
int CenterOfMassMultiMappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< CenterOfMassMultiMapping< MultiMapping<  MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes>  > > >()
        .add< CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > >()
        .add< CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState< Rigid3dTypes>, MechanicalState< Rigid3dTypes> > > >()
        .add< CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState< Rigid3dTypes>, MechanicalState< Vec3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CenterOfMassMultiMapping< MultiMapping<  MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes>  > > >()
        .add< CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > > >()
        .add< CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState< Rigid3fTypes>, MechanicalState< Rigid3fTypes> > > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MultiMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > ;
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > > ;
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > ;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MultiMapping< MechanicalState<Vec3fTypes>, MechanicalState< Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > ;
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping< MechanicalMultiMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > > ;
#endif

}
}
}
