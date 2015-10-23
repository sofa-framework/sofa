#define ZONEGENERATORIMAGETOOLBOX_CPP

#include <sofa/core/ObjectFactory.h>

#include "zonegeneratorimagetoolbox.h"
#include <image/ImageTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(ZoneGeneratorImageToolBox)

int ZoneGeneratorImageToolBox_Class = core::RegisterObject("ZoneGeneratorImageToolBox")
        .add<ZoneGeneratorImageToolBox<ImageUC> >()
        .add<ZoneGeneratorImageToolBox<ImageD> >(true)
#ifdef BUILD_ALL_IMAGE_TYPES
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
        .addAuthor("Vincent Majorczyk");

template<>
double ZoneGeneratorImageToolBox<ImageD>::color(int index,int max)
{
    return (double)index / (double)max;
}


template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox<ImageD>;



#ifdef BUILD_ALL_IMAGE_TYPES
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
