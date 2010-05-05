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
#ifndef SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_H

#include <sofa/helper/system/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/container/SpatialGridContainer.h>
#include <sofa/helper/rmath.h>
#include <vector>
#include <math.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::component::container;

template<class DataTypes>
class ParticlesRepulsionForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticlesRepulsionForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef sofa::core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

public:
    Data< Real > distance;
    Data< Real > stiffness;
    Data< Real > damping;

    typedef SpatialGridContainer<DataTypes> Grid;

    Grid* grid;

protected:

    struct Particle
    {
        sofa::helper::vector< int > neighbors; ///< indice + r/h
    };

    sofa::helper::vector<Particle> particles;

public:
    /// this method is called by the SpatialGrid when w connection between two particles is detected
    void addNeighbor(int i1, int i2, Real /*r2*/, Real /*h2*/)
    {
        //Real r_h = (Real)sqrt(r2/h2);
        if (i1<i2)
            particles[i1].neighbors.push_back(i2);
        else
            particles[i2].neighbors.push_back(i1);
    }

    ParticlesRepulsionForceField();

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

    virtual double getPotentialEnergy(const VecCoord& x) const;

    void draw();
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
