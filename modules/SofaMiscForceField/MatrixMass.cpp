/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MASS_MATRIXMASS_CPP
#include <SofaMiscForceField/MatrixMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;



// Register in the Factory
int MatrixMassClass = core::RegisterObject("Define a specific mass for each particle")
//.addAlias("MatrixMass")
// .add< MatrixMass<Vec3dTypes,double> >()
// .add< MatrixMass<Vec3fTypes,float> >()
// .add< MatrixMass<Vec2dTypes,double> >()
// .add< MatrixMass<Vec2fTypes,float> >()
        .add< MatrixMass<Vec3Types,Mat3x3d> >()
        .add< MatrixMass<Vec2Types,Mat2x2d> >()
        .add< MatrixMass<Vec1Types,Mat1x1d> >()

        ;

template class MatrixMass<Vec3Types,Mat3x3d>;
template class MatrixMass<Vec2Types,Mat2x2d>;
template class MatrixMass<Vec1Types,Mat1x1d>;


} // namespace mass

} // namespace component

} // namespace sofa

