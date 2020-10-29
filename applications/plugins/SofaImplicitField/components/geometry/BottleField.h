/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include "ScalarField.h"
#include <sofa/defaulttype/Mat.h>
namespace sofa::component::geometry
{

namespace _BottleField_
{

using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Mat3x3;

/**
 * This component emulates an implicit field shaped by a sphere with a hole made by an ellispsoid. The result may look like some kind of bottle or vase.
*/

class  SOFA_SOFAIMPLICITFIELD_API BottleField  : public ScalarField
{
public:
    SOFA_CLASS(BottleField, ScalarField);

public:
    BottleField() ;
    ~BottleField() override { }

    /// Inherited from BaseObject
    void init() override ;
    void reinit() override ;

    /// Inherited from ScalarField.
    double getValue(Vec3d& Pos, int &domain) override ;
    Vec3d getGradient(Vec3d &Pos, int& domain) override ;
    void getHessian(Vec3d &Pos, Mat3x3& h) override;

    double outerLength(Vec3d& Pos);
    double innerLength(Vec3d& Pos);

    using ScalarField::getValue ;
    using ScalarField::getGradient ;
    using ScalarField::getValueAndGradient ;

    Data<bool> d_inside; ///< If true the field is oriented inside (resp. outside) the sphere. (default = false)
    Data<double> d_radiusSphere; ///< Radius of Sphere emitting the field. (default = 1)
    Data<Vec3d> d_centerSphere; ///< Position of the Sphere Surface. (default=0 0 0)
    Data<double> d_shift;
    Data<double> d_ellipsoidRadius;
    Data<double> d_excentricity;
protected:
    Vec3d m_center;
    double m_radius;
    bool m_inside;
    double m_shift;
    double m_ellipsoidRadius;
    double m_excentricity;
};

} //namespace _BottleField_

using sofa::component::geometry::_BottleField_::BottleField;

} //namespace sofa::component::geometry

