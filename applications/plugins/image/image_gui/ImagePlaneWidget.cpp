/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_IMAGE_IMAGEPLANEWIDGET_CPP

#include "ImagePlaneWidget.h"
#include <sofa/helper/Factory.h>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

template class SOFA_IMAGE_GUI_API TDataWidget<ImageUC>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API TDataWidget<ImageC>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageI>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageUI>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageS>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageUS>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageL>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageUL>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageF>;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageB>;
#endif

template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<unsigned char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<double> >;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<unsigned int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<unsigned short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<unsigned long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<float> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImagePlane<bool> >;
#endif

helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned char> > >	DWClass_imagepUC("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<double> > >		DWClass_imagepD("imageplane",true);
#ifdef BUILD_ALL_IMAGE_TYPES
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<char> > >		DWClass_imagepC("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<int> > >		DWClass_imagepI("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned int> > >	DWClass_imagepUI("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<short> > >		DWClass_imagepS("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned short> > >	DWClass_imagepUS("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<long> > >		DWClass_imagepL("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned long> > >	DWClass_imagepUL("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<float> > >		DWClass_imagepF("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<bool> > >		DWClass_imagepB("imageplane",true);
#endif

} // qt
} // gui
} // sofa


