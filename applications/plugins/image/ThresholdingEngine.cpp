#include "ThresholdingEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ThresholdingEngine)

int ThresholdingEngineClass = core::RegisterObject("Find image threshold")
        .add<ThresholdingEngine<ImageUC>>(true)
        .add<ThresholdingEngine<ImageC>>()
        .add<ThresholdingEngine<ImageUS>>()
        .add<ThresholdingEngine<ImageS>>()
        .add<ThresholdingEngine<ImageUI>>()
        .add<ThresholdingEngine<ImageI>>()
        .add<ThresholdingEngine<ImageUL>>()
        .add<ThresholdingEngine<ImageL>>()
        .add<ThresholdingEngine<ImageF>>()
        .add<ThresholdingEngine<ImageD>>()
        .add<ThresholdingEngine<ImageB>>()
        ;

template class SOFA_IMAGE_API ThresholdingEngine<ImageUC>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageC>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageS>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageUS>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageI>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageUI>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageL>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageUL>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageF>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageD>;
template class SOFA_IMAGE_API ThresholdingEngine<ImageB>;


} // namespace engine
} // namespace component
} // namespace sofa

