#define FLEXIBLE_COMPUTEDUALQUATENGINE_CPP

#include "../initFlexible.h"

#include "ComputeDualQuatEngine.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{



SOFA_DECL_CLASS( ComputeDualQuatEngine )

using namespace defaulttype;

int ComputeDualQuatEngineClass = core::RegisterObject("Converts a vector of Affine or Rigid to a vector of Dual Quaternions.")
    .add< ComputeDualQuatEngine<Rigid3Types> >()
    .add< ComputeDualQuatEngine<Affine3Types> >()
;


template class SOFA_Flexible_API ComputeDualQuatEngine<Rigid3Types>;
template class SOFA_Flexible_API ComputeDualQuatEngine<Affine3Types>;


} // namespace engine

} // namespace component

} // namespace sofa
