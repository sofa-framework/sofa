#define LABELGRIDIMAGETOOLBOX_CPP

#include <sofa/core/ObjectFactory.h>

#include "labelgridimagetoolbox.h"
#include <image/ImageTypes.h>




namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(LabelGridImageToolBox)

int LabelGridImageToolBox_Class = core::RegisterObject("LabelGridImageToolBox")
        .add<LabelGridImageToolBox<ImageUC> >()
        .add<LabelGridImageToolBox<ImageD> >(true)
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<LabelGridImageToolBox<ImageC> >()
        .add<LabelGridImageToolBox<ImageI> >()
        .add<LabelGridImageToolBox<ImageUI> >()
        .add<LabelGridImageToolBox<ImageS> >()
        .add<LabelGridImageToolBox<ImageUS> >()
        .add<LabelGridImageToolBox<ImageL> >()
        .add<LabelGridImageToolBox<ImageUL> >()
        .add<LabelGridImageToolBox<ImageF> >()
        .add<LabelGridImageToolBox<ImageB> >()
#endif
        .addLicense("LGPL")
        .addAuthor("Vincent Majorczyk");

template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageC>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageI>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageUI>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageS>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageUS>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageL>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageUL>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageF>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageB>;
#endif

}}}



