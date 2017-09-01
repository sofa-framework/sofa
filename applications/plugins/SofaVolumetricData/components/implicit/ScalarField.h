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
#ifndef SOFAVOLUMETRICDATA_IMPLICIT_SCALARFIELD_H
#define SOFAVOLUMETRICDATA_IMPLICIT_SCALARFIELD_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace component
{

namespace implicit
{

using sofa::core::objectmodel::BaseObject ;
using sofa::defaulttype::Vector3 ;

class ScalarField : public BaseObject {

public:
    ScalarField() { }
    virtual ~ScalarField() { }
    virtual double eval(Vector3 p) = 0;
};

}

using implicit::ScalarField ;

} /// component

} /// sofa

#endif // IMPLICIT_SHAPE
