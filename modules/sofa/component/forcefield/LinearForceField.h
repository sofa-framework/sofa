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
#ifndef SOFA_COMPONENT_LINEARFORCEFIELD_H
#define SOFA_COMPONENT_LINEARFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
// #include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/component/topology/PointSubset.h>

namespace sofa
{

namespace component
{

namespace forcefield
{
/** Apply forces changing to given degres of freedom. Some keyTimes are given
 * and the force to be applied is linearly interpolated between keyTimes. */
template<class DataTypes>
class LinearForceField : public core::componentmodel::behavior::ForceField<DataTypes>
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef topology::PointSubset VecIndex;

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

    void draw();

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

    // ForceField methods
    /// Add the forces
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    /// Compute the force derivative
    virtual void addDForce (VecDeriv& , const VecDeriv& , const VecDeriv&);

    virtual double getPotentialEnergy(const VecCoord& x);

private :
    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;

    /// the forces corresponding to the surrounding key times
    Deriv prevF, nextF;

    /// initial constrained DOFs position
    VecCoord x0;

}; // definition of the LinearForceField class


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
