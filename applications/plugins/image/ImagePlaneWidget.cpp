/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_IMAGE_IMAGEPLANEWIDGET_CPP

#include "ImagePlaneWidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

template class SOFA_IMAGE_API TDataWidget<ImageC>;
template class SOFA_IMAGE_API TDataWidget<ImageUC>;
template class SOFA_IMAGE_API TDataWidget<ImageI>;
template class SOFA_IMAGE_API TDataWidget<ImageUI>;
template class SOFA_IMAGE_API TDataWidget<ImageS>;
template class SOFA_IMAGE_API TDataWidget<ImageUS>;
template class SOFA_IMAGE_API TDataWidget<ImageL>;
template class SOFA_IMAGE_API TDataWidget<ImageUL>;
template class SOFA_IMAGE_API TDataWidget<ImageF>;
template class SOFA_IMAGE_API TDataWidget<ImageD>;
template class SOFA_IMAGE_API TDataWidget<ImageB>;

//template class SOFA_IMAGE_API TDataWidget<ImagePlane<char> >;
template class SOFA_IMAGE_API TDataWidget<ImagePlane<unsigned char> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<int> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<unsigned int> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<short> >;
template class SOFA_IMAGE_API TDataWidget<ImagePlane<unsigned short> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<long> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<unsigned long> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<float> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<double> >;
//template class SOFA_IMAGE_API TDataWidget<ImagePlane<bool> >;

//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<char> > >			DWClass_imagepC("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned char> > >	DWClass_imagepUC("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<int> > >			DWClass_imagepI("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned int> > >	DWClass_imagepUI("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<short> > >			DWClass_imagepS("imageplane",true);
helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned short> > >	DWClass_imagepUS("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<long> > >			DWClass_imagepL("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<unsigned long> > >	DWClass_imagepUL("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<float> > >			DWClass_imagepF("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<double> > >			DWClass_imagepD("imageplane",true);
//helper::Creator<DataWidgetFactory, ImagePlaneDataWidget< ImagePlane<bool> > >			DWClass_imagepB("imageplane",true);


} // qt
} // gui
} // sofa


