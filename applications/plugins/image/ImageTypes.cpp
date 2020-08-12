#include <sofa/defaulttype/TemplatesAliases.h>
#include <sofa/core/ObjectFactory.h>

#include <image/config.h>
#include "ImageTypes.h"
#ifdef WITH_MULTITHREADING
#include <MultiThreading/src/DataExchange.inl>
#endif  // WITH_MULTITHREADING

namespace sofa
{
namespace defaulttype
{


static RegisterTemplateAlias ImageRAlias("ImageR", "ImageD");


#ifdef WITH_MULTITHREADING
// Register in the Factory
using sofa::core::DataExchange;
int DataExchangeClass = core::RegisterObject("DataExchange")
.add< DataExchange<sofa::defaulttype::ImageB>>()
.add< DataExchange<sofa::defaulttype::ImageC>>()
.add< DataExchange<sofa::defaulttype::ImageUC>>()
.add< DataExchange<sofa::defaulttype::ImageI>>()
.add< DataExchange<sofa::defaulttype::ImageUI>>()
.add< DataExchange<sofa::defaulttype::ImageS>>()
.add< DataExchange<sofa::defaulttype::ImageUS>>()
.add< DataExchange<sofa::defaulttype::ImageL>>()
.add< DataExchange<sofa::defaulttype::ImageUL>>()
.add< DataExchange<sofa::defaulttype::ImageF>>()
.add< DataExchange<sofa::defaulttype::ImageD>>()
;
}  // namespace defaulttype

namespace core
{
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageB>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageC>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageUC>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageI>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageUI>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageS>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageUS>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageL>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageUL>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageF>;
template class SOFA_IMAGE_API core::DataExchange<sofa::defaulttype::ImageD>;
#endif  // WITH_MULTITHREADING
}
}  // namespace sofa
