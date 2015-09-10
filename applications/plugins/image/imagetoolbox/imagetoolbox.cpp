#define SOFA_IMAGE_IMAGETOOLBOX_CPP

#include <QDataStream>
#include <QMetaType>
#include "imagetoolbox.h"
#include <sofa/core/ObjectFactory.h>
#include <image/image_gui/config.h>


namespace sofa
{
namespace component
{
namespace misc
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS (ImageToolBox);
// Register in the Factory

int ImageToolBoxClass = core::RegisterObject ( "ImageToolBox" )
        .add<ImageToolBox<ImageUC> >(true)
        .add<ImageToolBox<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
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
#ifdef BUILD_ALL_IMAGE_TYPES
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
