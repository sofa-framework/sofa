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
#ifndef SOFA_COMPONENT_FORCEFIELD_ASPIRATEFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_ASPIRATEFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>

#include <sofa/component/component.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// Apply constant forces to given degrees of freedom.
template<class DataTypes>
class AspirationForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AspirationForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef defaulttype::RigidCoord<3,double> RigidCoord;
    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef topology::PointSubset VecIndex;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

public:

    Data< Deriv > f_force;
    Data< double > f_scale;
    Data< double > f_arrowSizeCoef; // for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< double > f_ray; // for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data<RigidCoord > f_positionRigid;
    Data< double > f_opposite_pressure;

    AspirationForceField();

    /// Add the forces
    virtual void addForce (const core::MechanicalParams* params /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    /// Constant force has null variation
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */);

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset);

    /// Constant force has null variation
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/, double /*kFact*/) {}

    void draw(const core::visual::VisualParams*);
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_ASPIRATEFORCEFIELD_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API AspirationForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API AspirationForceField<Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_H
