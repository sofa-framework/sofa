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
#include <sofa/type/Mat.h>

namespace sofa::component::geometry
{

namespace _StarShapedField_
{

using sofa::type::Vec3d;
using sofa::type::Mat3x3;

/**
 * This component emulates an implicit field that looks like some kind of star.
*/
class  SOFA_SOFAIMPLICITFIELD_API StarShapedField  : public ScalarField
{
public:
    SOFA_CLASS(StarShapedField, ScalarField);

public:
    StarShapedField() ;
    ~StarShapedField() override { }

    /// Inherited from BaseObject
    void init() override ;
    void reinit() override ;

    /// Inherited from ScalarField.
    double getValue(Vec3d& Pos, int &domain) override ;
    Vec3d getGradient(Vec3d &Pos, int& domain) override ;
    void getHessian(Vec3d &Pos, Mat3x3& h) override;

    using ScalarField::getValue ;
    using ScalarField::getGradient ;
    using ScalarField::getValueAndGradient ;

    Data<bool> d_inside; ///< If true the field is oriented inside (resp. outside) the sphere. (default = false)
    Data<double> d_radiusSphere; ///< Radius of Sphere emitting the field. (default = 1)
    Data<Vec3d> d_centerSphere; ///< Position of the Sphere Surface. (default=0 0 0)
    Data<double> d_branches; ///< Number of branches of the star. (default=1)
    Data<double> d_branchesRadius; ///< Size of the branches of the star. (default=1)
protected:
    Vec3d m_center;
    double m_radius;
    bool m_inside;
    double m_branches;
    double m_branchesRadius;
};

} // namespace _StarShapedField_

using sofa::component::geometry::_StarShapedField_::StarShapedField;

} // namespace sofa::component::geometry

