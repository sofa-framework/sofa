#define DISTANCEZONEIMAGETOOLBOX_CPP

#include "distancezoneimagetoolbox.h"
#include <sofa/core/ObjectFactory.h>

#include <image/ImageTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

void registerDistanceZoneImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("DistanceZoneImageToolBox")
    .add<DistanceZoneImageToolBox<ImageUC> >()
    .add<DistanceZoneImageToolBox<ImageD> >(true)
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
    .add<DistanceZoneImageToolBox<ImageC> >()
    .add<DistanceZoneImageToolBox<ImageI> >()
    .add<DistanceZoneImageToolBox<ImageUI> >()
    .add<DistanceZoneImageToolBox<ImageS> >()
    .add<DistanceZoneImageToolBox<ImageUS> >()
    .add<DistanceZoneImageToolBox<ImageL> >()
    .add<DistanceZoneImageToolBox<ImageUL> >()
    .add<DistanceZoneImageToolBox<ImageF> >()
    .add<DistanceZoneImageToolBox<ImageB> >()
#endif
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}



template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageD>;



#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageC>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageI>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageUI>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageS>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageUS>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageL>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageUL>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageF>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageB>;

#endif






}}}
