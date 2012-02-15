#define SOFA_IMAGE_VECTORVISWIDGET_CPP

#include "VectorVisualizationWidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

SOFA_DECL_CLASS(VectorVisualizationDataWidget);

template class SOFA_IMAGE_API TDataWidget<VectorVis>;

/**
* Defines that type of Data the VectorVisualizationWidget can communicate with
*/
helper::Creator<DataWidgetFactory, VectorVisualizationDataWidget< VectorVis > >	DWClass_VectorVis("vectorvis",true);

} // qt
} // gui
} // sofa
