#define SOFA_IMAGE_IMAGETOOLBOX_CPP

#include "imagetoolbox.h"

#include <QDataStream>
#include <QMetaType>

#include <sofa/core/ObjectFactory.h>
#include <image_gui/config.h>


namespace sofa
{
namespace component
{
namespace misc
{
using namespace sofa::defaulttype;

// Register in the Factory

int ImageToolBoxClass = core::RegisterObject ( "ImageToolBox" )
        .add<ImageToolBox<ImageUC> >(true)
        .add<ImageToolBox<ImageD> >()
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
        .add<ImageToolBox<ImageC> >()
        .add<ImageToolBox<ImageI> >()
        .add<ImageToolBox<ImageUI> >()
        .add<ImageToolBox<ImageS> >()
        .add<ImageToolBox<ImageUS> >()
        .add<ImageToolBox<ImageL> >()
        .add<ImageToolBox<ImageUL> >()
        .add<ImageToolBox<ImageF> >()
        .add<ImageToolBox<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_GUI_API ImageToolBox<ImageUC>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageD>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageC>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageI>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageUI>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageS>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageUS>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageL>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageUL>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageF>;
template class SOFA_IMAGE_GUI_API ImageToolBox<ImageB>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
