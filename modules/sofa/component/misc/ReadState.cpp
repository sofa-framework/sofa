#include <sofa/component/misc/ReadState.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ReadState)

using namespace defaulttype;

int ReadStateClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< ReadState >();


} // namespace misc

} // namespace component

} // namespace sofa
