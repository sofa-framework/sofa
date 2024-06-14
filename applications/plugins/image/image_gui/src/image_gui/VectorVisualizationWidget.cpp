#define SOFA_IMAGE_VECTORVISWIDGET_CPP

#include <image_gui/VectorVisualizationWidget.h>
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

template class SOFA_IMAGE_GUI_API TDataWidget<VectorVis>;

/**
* Defines that type of Data the VectorVisualizationWidget can communicate with
*/
helper::Creator<DataWidgetFactory, VectorVisualizationDataWidget< VectorVis > >	DWClass_VectorVis("vectorvis",true);

} // qt
} // gui
} // sofa
