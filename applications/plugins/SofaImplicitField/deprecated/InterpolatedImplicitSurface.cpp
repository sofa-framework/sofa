#include <SofaImplicitField/config.h>
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include <SofaImplicitField/deprecated/InterpolatedImplicitSurface.h>

namespace sofa::component::container
{

// Register in the Factory
void registerInterpolatedImplicitSurface(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Deprecated. This class is forwarding DiscreteGridField.")
    .add< InterpolatedImplicitSurface >()
    .addAlias("DistGrid"));
}

} /// sofa::component::container
