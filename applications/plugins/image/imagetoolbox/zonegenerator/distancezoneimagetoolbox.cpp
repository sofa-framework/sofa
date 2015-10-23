#define DISTANCEZONEIMAGETOOLBOX_CPP

#include <sofa/core/ObjectFactory.h>

#include "distancezoneimagetoolbox.h"
#include <image/ImageTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(DistanceZoneImageToolBox)

int DistanceZoneImageToolBox_Class = core::RegisterObject("DistanceZoneImageToolBox")
        .add<DistanceZoneImageToolBox<ImageUC> >()
        .add<DistanceZoneImageToolBox<ImageD> >(true)
#ifdef BUILD_ALL_IMAGE_TYPES
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
        .addAuthor("Vincent Majorczyk");




template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox<ImageD>;



#ifdef BUILD_ALL_IMAGE_TYPES
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
