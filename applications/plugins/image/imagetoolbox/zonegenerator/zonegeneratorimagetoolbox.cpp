#define ZONEGENERATORIMAGETOOLBOX_CPP

#include "zonegeneratorimagetoolbox.h"
#include <sofa/core/ObjectFactory.h>

#include <image/ImageTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

void registerZoneGeneratorImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("ZoneGeneratorImageToolBox")
    .add<ZoneGeneratorImageToolBox<ImageUC> >()
    .add<ZoneGeneratorImageToolBox<ImageD> >(true)
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
    .add<ZoneGeneratorImageToolBox<ImageC> >()
    .add<ZoneGeneratorImageToolBox<ImageI> >()
    .add<ZoneGeneratorImageToolBox<ImageUI> >()
    .add<ZoneGeneratorImageToolBox<ImageS> >()
    .add<ZoneGeneratorImageToolBox<ImageUS> >()
    .add<ZoneGeneratorImageToolBox<ImageL> >()
    .add<ZoneGeneratorImageToolBox<ImageUL> >()
    .add<ZoneGeneratorImageToolBox<ImageF> >()
    .add<ZoneGeneratorImageToolBox<ImageB> >()
#endif
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}


template<>
double ZoneGeneratorImageToolBox<ImageD>::color(int index,int max)
{
    return (double)index / (double)max;
}


template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageD>;



#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageC>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageI>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageUI>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageS>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageUS>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageL>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageUL>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageF>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageB>;

#endif






}}}
