/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_IMAGE_HISTOGRAMWIDGET_CPP

#include "HistogramWidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

SOFA_DECL_CLASS(HistogramDataWidget);

template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<unsigned char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<double> >;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<char> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<unsigned int> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<unsigned short> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<unsigned long> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<float> >;
template class SOFA_IMAGE_GUI_API TDataWidget<Histogram<bool> >;
#endif

helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<unsigned char> > >	DWClass_histoUC("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<double> > >			DWClass_histoD("imagehistogram",true);
#ifdef BUILD_ALL_IMAGE_TYPES
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<char> > >				DWClass_histoC("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<int> > >				DWClass_histoI("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<unsigned int> > >		DWClass_histoUI("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<short> > >			DWClass_histoS("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<unsigned short> > >	DWClass_histoUS("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<long> > >				DWClass_histoL("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<unsigned long> > >	DWClass_histoUL("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<float> > >			DWClass_histoF("imagehistogram",true);
helper::Creator<DataWidgetFactory, HistogramDataWidget< Histogram<bool> > >				DWClass_histoB("imagehistogram",true);
#endif

} // qt
} // gui
} // sofa


