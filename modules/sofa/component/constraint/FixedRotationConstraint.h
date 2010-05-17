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
#ifndef SOFA_COMPONENT_CONSTRAINT_FixedRotationConstraint_H
#define SOFA_COMPONENT_CONSTRAINT_FixedRotationConstraint_H

#include <sofa/core/behavior/Constraint.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace component
{

namespace constraint
{


using namespace sofa::helper;
using namespace sofa::defaulttype;


/** Prevents rotation around X or Y or Z axis
*/
template <class DataTypes>
class FixedRotationConstraint : public core::behavior::Constraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedRotationConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::Constraint, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Vec<3,Real> Vec3;

public:

    FixedRotationConstraint();
    virtual ~FixedRotationConstraint();

    void init();
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx);

    void projectResponse(VecDeriv& dx);
    void projectResponse(SparseVecDeriv& dx);
    virtual void projectVelocity(VecDeriv& dx); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x); ///< project x to constrained space (x models a position)

    virtual void draw();


protected :
    Data< bool > FixedXRotation;
    Data< bool > FixedYRotation;
    Data< bool > FixedZRotation;
    vector<Quat> previousOrientation;
};

} // namespace constraint

} // namespace component

} // namespace sofa


#endif
