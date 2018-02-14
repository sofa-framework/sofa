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
#ifndef SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_INL

#include "LinearForceField.h"
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/BaseVector.h>

#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
LinearForceField<DataTypes>::LinearForceField()
    : data(new LinearForceFieldInternalData<DataTypes>())
    , points(initData(&points, "points", "points where the force is applied"))
    , d_force(initData(&d_force, (Real)1.0, "force", "applied force to all points"))
    , d_keyTimes(initData(&d_keyTimes, "times", "key times for the interpolation"))
    , d_keyForces(initData(&d_keyForces, "forces", "forces corresponding to the key times"))
    , d_arrowSizeCoef(initData(&d_arrowSizeCoef,(SReal)0.0, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
{ }


template<class DataTypes>
void LinearForceField<DataTypes>::init()
{
    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters for topology data and handler
    points.createTopologicalEngine(topology);
    points.registerTopologicalData();

    Inherit::init();
}

template<class DataTypes>
void LinearForceField<DataTypes>::addPoint(unsigned index)
{
    points.beginEdit()->push_back(index);
    points.endEdit();

}// LinearForceField::addPoint

template<class DataTypes>
void LinearForceField<DataTypes>::removePoint(unsigned /*index*/)
{
// removeValue(*points.beginEdit(), index);
    //points.endEdit();

}// LinearForceField::removePoint

template<class DataTypes>
void LinearForceField<DataTypes>::clearPoints()
{
    points.beginEdit()->clear();
    points.endEdit();

}// LinearForceField::clearPoints


template<class DataTypes>
void LinearForceField<DataTypes>::addKeyForce(Real time, Deriv force)
{
    // TODO : sort the key force while adding a new one
    d_keyTimes.beginEdit()->push_back( time);
    d_keyTimes.endEdit();
    d_keyForces.beginEdit()->push_back( force );
    d_keyForces.endEdit();

}// LinearForceField::addKeyForce

template<class DataTypes>
void LinearForceField<DataTypes>::clearKeyForces()
{
    d_keyTimes.beginEdit()->clear();
    d_keyTimes.endEdit();
    d_keyForces.beginEdit()->clear();
    d_keyForces.endEdit();

}// LinearForceField::clearKeyForces

template<class DataTypes>
void LinearForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f1, const DataVecCoord& /*p1*/, const DataVecDeriv&)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > _f1 = f1;

    Real cT = (Real) this->getContext()->getTime();

    if (d_keyTimes.getValue().size() != 0 && cT >= *d_keyTimes.getValue().begin())
    {
        Deriv targetForce;
        if (cT < *d_keyTimes.getValue().rbegin())
        {
            nextT = *d_keyTimes.getValue().begin();
            prevT = nextT;

            bool finished = false;

            typename helper::vector< Real >::const_iterator it_t = d_keyTimes.getValue().begin();
            typename VecDeriv::const_iterator it_f = d_keyForces.getValue().begin();

            // WARNING : we assume that the key-events are in chronological order
            // here we search between which keyTimes we are.
            while( it_t != d_keyTimes.getValue().end() && !finished)
            {
                if ( *it_t <= cT )
                {
                    prevT = *it_t;
                    prevF = *it_f;
                }
                else
                {
                    nextT = *it_t;
                    nextF = *it_f;
                    finished = true;
                }
                it_t++;
                it_f++;
            }
            if (finished)
            {
                Deriv slope = (nextF - prevF)*(1.0/(nextT - prevT));
                targetForce = slope*(cT - prevT) + prevF;
                targetForce *= d_force.getValue();
            }
        }
        else
        {
            targetForce = d_keyForces.getValue()[d_keyTimes.getValue().size() - 1];
        }

        for(auto index : points.getValue())
        {
            _f1[index] += targetForce;
        }
    }
}// LinearForceField::addForce

template<class DataTypes>
void LinearForceField<DataTypes>::addKToMatrix(defaulttype::BaseMatrix* matrix, SReal kFact, unsigned int& offset)
{
    SOFA_UNUSED(matrix);
    SOFA_UNUSED(kFact);
    SOFA_UNUSED(offset);
}

template<class DataTypes>
SReal LinearForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    Real cT = (Real) this->getContext()->getTime();
    const VecCoord& _x = x.getValue();
    const SetIndexArray& indices = points.getValue();
    SReal e=0;
    if (d_keyTimes.getValue().size() != 0 && cT >= *d_keyTimes.getValue().begin() && cT <= *d_keyTimes.getValue().rbegin() && prevT != nextT)
    {
        Real dt = (cT - prevT)/(nextT - prevT);
        Deriv ff = (nextF - prevF)*dt + prevF;

        Real f = d_force.getValue();

        for(unsigned i = 0; i < indices.size(); i++)
        {
            e -= ff*_x[i]*f;
        }
    }

    return e;
}// LinearForceField::getPotentialEnergy

template< class DataTypes>
void LinearForceField<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{


}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_INL
