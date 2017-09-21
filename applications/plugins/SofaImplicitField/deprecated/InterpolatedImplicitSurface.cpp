#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include <SofaImplicitField/deprecated/InterpolatedImplicitSurface.h>

namespace sofa
{
namespace component
{
namespace container
{

SOFA_DECL_CLASS(InterpolatedImplicitSurface)

///factory register
int InterpolatedImplicitSurfaceClass = RegisterObject("Deprecated. This class is forwarding DiscreteGridField.")
        .add< InterpolatedImplicitSurface >()
        .addAlias("DistGrid") ;

} /// container
} /// component
} /// sofa
