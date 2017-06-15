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

