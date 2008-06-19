
#include <sofa/component/misc/WriteState.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(WriteState)

using namespace defaulttype;



int WriteStateClass = core::RegisterObject("Write State vectors to file at each timestep")
        .add< WriteState >();

} // namespace misc

} // namespace component

} // namespace sofa
