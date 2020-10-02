#define SOFA_IMAGE_IMAGETOOLBOXWIDGET_CPP

#include "imagetoolboxwidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

//class SOFA_IMAGE_GUI_API imagetoolbox_data_widget_container;

template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<unsigned char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<double> >;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<unsigned int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<unsigned short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<unsigned long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<float> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageToolBoxData<bool> >;
#endif

helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<unsigned char> > >	DWClass_imagetoolboxUC("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<double> > >			DWClass_imagetoolboxD("ImageToolBoxWidget",true);
#ifdef BUILD_ALL_IMAGE_TYPES
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<char> > >				DWClass_imagetoolboxC("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<int> > >				DWClass_imagetoolboxI("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<unsigned int> > >		DWClass_imagetoolboxUI("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<short> > >			DWClass_imagetoolboxS("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<unsigned short> > >	DWClass_imagetoolboxUS("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<long> > >				DWClass_imagetoolboxL("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<unsigned long> > >	DWClass_imagetoolboxUL("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<float> > >			DWClass_imagetoolboxF("ImageToolBoxWidget",true);
helper::Creator<DataWidgetFactory, ImageToolBoxWidget< ImageToolBoxData<bool> > >				DWClass_imagetoolboxB("ImageToolBoxWidget",true);
#endif

} // qt
} // gui
} // sofa
