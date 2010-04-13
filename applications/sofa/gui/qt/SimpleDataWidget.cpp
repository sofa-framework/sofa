/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/gui/qt/SimpleDataWidget.h>
#include <sofa/helper/Factory.inl>
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Creator;
using sofa::helper::fixed_array;
using namespace sofa::defaulttype;
using sofa::component::fem::QuadraturePoint;
using sofa::helper::Polynomial_LD;

SOFA_DECL_CLASS(SimpleDataWidget);

Creator<DataWidgetFactory, SimpleDataWidget<bool> > DWClass_bool("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<char> > DWClass_char("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<unsigned char> > DWClass_uchar("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<int> > DWClass_int("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<unsigned int> > DWClass_uint("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<float> > DWClass_float("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<double> > DWClass_double("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<std::string> > DWClass_string("default",true);

//Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,5> > >DWClass_PolynomialLD5d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,4> > >DWClass_PolynomialLD4d("default",true);
//<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,3> > >DWClass_PolynomialLD3d("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,2> > >DWClass_PolynomialLD2d("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float, 5> > >DWClass_PolynomialLD5f("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,int> > > DWClass_Vec1i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,unsigned int> > > DWClass_Vec1u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,float> > > DWClass_Vec1f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,double> > > DWClass_Vec1d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,int> > > DWClass_Vec2i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,unsigned int> > > DWClass_Vec2u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,float> > > DWClass_Vec2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,double> > > DWClass_Vec2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,int> > > DWClass_Vec3i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,unsigned int> > > DWClass_Vec3u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,float> > > DWClass_Vec3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,double> > > DWClass_Vec3d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,int> > > DWClass_Vec4i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,unsigned int> > > DWClass_Vec4u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,float> > > DWClass_Vec4f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,double> > > DWClass_Vec4d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,int> > > DWClass_Vec6i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,unsigned int> > > DWClass_Vec6u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,float> > > DWClass_Vec6f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,double> > > DWClass_Vec6d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<8,int> > > DWClass_Vec8i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<8,unsigned int> > > DWClass_Vec8u("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,1> > > DWClass_fixed_array1i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,1> > > DWClass_fixed_array1u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,2> > > DWClass_fixed_array2i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,2> > > DWClass_fixed_array2u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,3> > > DWClass_fixed_array3i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,3> > > DWClass_fixed_array3u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,4> > > DWClass_fixed_array4i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,4> > > DWClass_fixed_array4u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,6> > > DWClass_fixed_array6i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,6> > > DWClass_fixed_array6u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,8> > > DWClass_fixed_array8i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,8> > > DWClass_fixed_array8u("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Quater<float> > > DWClass_Quatf("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Quater<double> > > DWClass_Quatd("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< sofa::component::fem::QuadraturePoint< Vec3f > > >DWClass_QPf("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::component::fem::QuadraturePoint< Vec3d > > >DWClass_QPd("default",true);



Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,2,float> > > DWClass_Mat22f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,2,double> > > DWClass_Mat22d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,3,float> > > DWClass_Mat23f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,3,double> > > DWClass_Mat23d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,3,float> > > DWClass_Mat33f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,3,double> > > DWClass_Mat33d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,4,float> > > DWClass_Mat34f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,4,double> > > DWClass_Mat34d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<4,4,float> > > DWClass_Mat44f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<4,4,double> > > DWClass_Mat44d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<6,6,float> > > DWClass_Mat66f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<6,6,double> > > DWClass_Mat66d("default",true);



} // namespace qt

} // namespace gui

} // namespace sofa
