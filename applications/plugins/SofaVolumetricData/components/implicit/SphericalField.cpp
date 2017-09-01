/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
/******************************************************************************
* Contributors:                                                               *
*   - thomas.goss@etudiant.univ-lille1.fr                                     *
*   - damien.marchal@univ-lille1.fr                                           *
******************************************************************************/
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include "SphericalField.h"

namespace sofa
{

namespace component
{

namespace implicit
{

double SphericalField::eval(Vector3 p) {

    double x=p.x(), y=p.y(), z=p.z();
    double x2=x*x, y2=y*y, z2=z*z;
    double x4=x2*x2, y4=y2*y2, z4=z2*z2;
    return x4  + y4  + z4  + 2 *x2*  y2  + 2* x2*z2  + 2*y2*  z2  - 5 *x2  + 4* y2  - 5*z2+4;
}


///factory register
int SphericalFieldComponent = RegisterObject("Implement a spherical distance field function").add< SphericalField >();

} /// implicit
} /// component
} /// sofa
