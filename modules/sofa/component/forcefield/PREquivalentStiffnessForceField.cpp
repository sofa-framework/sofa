#include <sofa/component/forcefield/PREquivalentStiffnessForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(PREquivalentStiffnessForceField)

int PREquivalentStiffnessForceFieldClass = core::RegisterObject("Partial Rigidification equivalent stiffness forcefield")
#ifndef SOFA_FLOAT
    .add<PREquivalentStiffnessForceField<sofa::defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
    .add<PREquivalentStiffnessForceField<sofa::defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
    template class PREquivalentStiffnessForceField<sofa::defaulttype::Rigid3dTypes>;
#endif

#ifndef SOFA_FLOAT
    template class PREquivalentStiffnessForceField<sofa::defaulttype::Rigid3fTypes>;
#endif

} // forcefield

} // component

} // sofa
