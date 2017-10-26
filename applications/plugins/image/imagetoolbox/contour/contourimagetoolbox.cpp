#define CONTOURIMAGETOOLBOX_CPP

#include "contourimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

#include <image/ImageTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(ContourImageToolBox)

int ContourImageToolBox_Class = core::RegisterObject("ContourImageToolBox")
        .add<ContourImageToolBox<ImageUC> >()
        .add<ContourImageToolBox<ImageD> >(true)
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ContourImageToolBox<ImageC> >()
        .add<ContourImageToolBox<ImageI> >()
        .add<ContourImageToolBox<ImageUI> >()
        .add<ContourImageToolBox<ImageS> >()
        .add<ContourImageToolBox<ImageUS> >()
        .add<ContourImageToolBox<ImageL> >()
        .add<ContourImageToolBox<ImageUL> >()
        .add<ContourImageToolBox<ImageF> >()
        .add<ContourImageToolBox<ImageB> >()
#endif
        .addLicense("LGPL")
        .addAuthor("Vincent Majorczyk");

template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageC>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageI>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageUI>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageS>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageUS>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageL>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageUL>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageF>;
template class SOFA_IMAGE_GUI_API ContourImageToolBox<ImageB>;
#endif





}}}
