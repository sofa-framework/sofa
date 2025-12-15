
#include <image_multithread/MultithreadImageTypes.h>

#include <sofa/core/ObjectFactory.h>

#include <MultiThreading/DataExchange.inl>

namespace image_multithread
{


using sofa::core::DataExchange;

// Register in the Factory
void registerDataExchange(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("DataExchange")
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
    );
}

}  // namespace image_multithread

namespace sofa::core
{
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageB>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageC>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageUC>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageI>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageUI>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageS>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageUS>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageL>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageUL>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageF>;
template class SOFA_IMAGE_API DataExchange<sofa::defaulttype::ImageD>;

}
