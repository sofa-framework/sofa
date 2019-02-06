#include "UniformConstraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{
namespace constraint
{


int UniformConstraintClass = sofa::core::RegisterObject("A constraint equation applied on all dofs.")

.add< UniformConstraint<sofa::defaulttype::Vec1Types> >()

;


template class UniformConstraint<sofa::defaulttype::Vec1Types>;


}

}