#include "MeanComputation.inl"

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

int MeanComputationEngineClass = core::RegisterObject("Compute the mean of the input elements")
        .add< MeanComputation<defaulttype::Vec3Types> >(true) // default template
        .add< MeanComputation<defaulttype::Vec1Types> >()
        .add< MeanComputation<defaulttype::Vec2Types> >()
        .add< MeanComputation<defaulttype::Rigid2Types> >()
        .add< MeanComputation<defaulttype::Rigid3Types> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa
