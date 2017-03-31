/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Contributions:                                                              *
*     - damien.marchal@univ-lille1.fr                                         *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_DEFAULTTYPE_VEC_CPP

#include <sofa/defaulttype/Vec.h>

namespace sofa
{
namespace defaulttype
{

 template class Vec<2,float>;
 template class Vec<2,int>;
 template class Vec<2,double>;
 template class Vec<2,unsigned>;

 template class Vec<3,float>;
 template class Vec<3,int>;
 template class Vec<3,double>;
 template class Vec<3,unsigned>;

 template class Vec<4,float>;
 template class Vec<4,int>;
 template class Vec<4,double>;
 template class Vec<4,unsigned>;

 template class Vec<6,float>;
 template class Vec<6,int>;
 template class Vec<6,double>;
 template class Vec<6,unsigned>;


} // namespace defaulttype
} // namespace sofa

