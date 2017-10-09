#include "UniformConstraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{
namespace constraint
{


int UniformConstraintClass = sofa::core::RegisterObject("A constraint equation applied on all dofs.")

#ifndef SOFA_FLOAT
.add< UniformConstraint<sofa::defaulttype::Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< UniformConstraint<sofa::defaulttype::Vec1fTypes> >()
#endif
;


#ifndef SOFA_FLOAT
template class UniformConstraint<sofa::defaulttype::Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class UniformConstraint<sofa::defaulttype::Vec1fTypes>;
#endif

}

}