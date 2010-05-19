#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP
#include <sofa/component/mapping/SubsetMultiMapping.inl>
#include <sofa/core/MultiMapping.inl>
#include <sofa/core/behavior/MechanicalMultiMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/behavior/MechanicalState.h>

using namespace sofa::core::behavior;
using namespace sofa::defaulttype;
namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(SubsetMultiMapping)

// Register in the Factory
int SubsetMultiMappingClass = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
#ifndef SOFA_FLOAT
        .add< SubsetMultiMapping<MultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > >()
        .add< SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > >  >()
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > >  >()
        .add< SubsetMultiMapping<MultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > >  >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > ;
template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > ;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > > ;
template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > > ;
#endif


}
}
}
