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
#ifndef SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/topology/PointSubset.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class LinearForceFieldInternalData
{
public:
};

/** Apply forces changing to given degres of freedom. Some keyTimes are given
* and the force to be applied is linearly interpolated between keyTimes. */
template<class DataTypes>
class LinearForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef topology::PointSubset VecIndex;

protected:

    LinearForceFieldInternalData<DataTypes> *data;
    friend class LinearForceFieldInternalData<DataTypes>;

public:
    /// concerned DOFs
    Data< VecIndex > points;

    /// applied force for all the points
    Data< Real > force;

    /// the key frames when the forces are defined by the user
    Data< helper::vector< Real > > keyTimes;

    /// forces corresponding to the key frames
    Data< VecDeriv > keyForces;

    /// for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< double > arrowSizeCoef;

    LinearForceField();

    void draw(const core::visual::VisualParams* vparams);

    /// methods to add/remove some indices, keyTimes, keyForces
    void addPoint(unsigned index);
    void removePoint(unsigned int index);
    void clearPoints();

    /**
    * Add a new key force.
    * Key force should be added in classified order.
    *
    * @param time  the simulation time you want to set a movement (in sec)
    * @param force the corresponding force
    */
    void addKeyForce(Real time, Deriv force);
    void clearKeyForces();

    virtual void init() { Inherit::init(); };

    // ForceField methods
    /// Add the forces
    virtual void addForce (const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    /// Compute the force derivative
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */)
    {
        //TODO: remove this line (avoid warning message) ...
        mparams->kFactor();
    };

    virtual double getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;

private :
    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;

    /// the forces corresponding to the surrounding key times
    Deriv prevF, nextF;

    /// initial constrained DOFs position
    VecCoord x0;

}; // definition of the LinearForceField class

using sofa::defaulttype::Vec6dTypes;
using sofa::defaulttype::Vec6fTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec1dTypes;
using sofa::defaulttype::Vec1fTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)
#pragma warning(disable : 4231)

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec3dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec2dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec1dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec6dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Rigid3dTypes>;
// template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec3fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec2fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec1fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Vec6fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Rigid3fTypes>;
// template class SOFA_COMPONENT_FORCEFIELD_API LinearForceField<Rigid2fTypes>;
#endif

#endif // defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H
