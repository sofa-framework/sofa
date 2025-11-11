#define LABELGRIDIMAGETOOLBOX_CPP

#include "labelgridimagetoolbox.h"
#include <sofa/core/ObjectFactory.h>

#include <image/ImageTypes.h>




namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

void registerLabelGridImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("LabelGridImageToolBox")
    .add<LabelGridImageToolBox<ImageUC> >()
    .add<LabelGridImageToolBox<ImageD> >(true)
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
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
        .addAuthor("Vincent Majorczyk")
    );
}

template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API LabelGridImageToolBox<ImageD>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
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



