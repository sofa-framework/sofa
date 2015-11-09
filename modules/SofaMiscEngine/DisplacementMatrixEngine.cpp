#define FLEXIBLE_DisplacementMatrixENGINE_CPP

#include "DisplacementMatrixEngine.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{



SOFA_DECL_CLASS( DisplacementMatrixEngine )

using namespace defaulttype;

int DisplacementMatrixEngineClass = core::RegisterObject("Converts a vector of Rigid to a vector of displacement matrices.")
    .add< DisplacementMatrixEngine<Rigid3Types> >()
;


template class SOFA_MISC_ENGINE_API DisplacementMatrixEngine<Rigid3Types>;


} // namespace engine

} // namespace component

} // namespace sofa
