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
#include <SofaGeneralObjectInteraction/RepulsiveSpringForceField.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::interactionforcefield
{



template <class DataTypes>
void RepulsiveSpringForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& f,
    const DataVecCoord& x, const DataVecDeriv& v)
{
    VecDeriv& _f = *sofa::helper::getWriteAccessor(f);
    auto _x = sofa::helper::getReadAccessor(x);
    auto _v = sofa::helper::getReadAccessor(v);

    const type::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());

    _f.resize(_x.size());
    for (unsigned int i=0; i<springs.size(); i++)
    {
        int a = springs[i].m1;
        int b = springs[i].m2;
        Coord u = _x[b]-_x[a];
        Real d = u.norm();
        if (d!=0 && d < springs[i].initpos)
        {
            Real inverseLength = 1.0f/d;
            u *= inverseLength;
            Real elongation = (Real)(d - springs[i].initpos);
            Deriv relativeVelocity = _v[b]-_v[a];
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = (Real)(springs[i].ks*elongation+springs[i].kd*elongationVelocity);
            Deriv force = u*forceIntensity;
            _f[a]+=force;
            _f[b]-=force;

            Mat& m = this->dfdx[i];
            Real tgt = forceIntensity * inverseLength;
            for( int j=0; j<N; ++j )
            {
                for( int k=0; k<N; ++k )
                {
                    m[j][k] = ((Real)springs[i].ks-tgt) * u[j] * u[k];
                }
                m[j][j] += tgt;
            }
        }
        else
        {
            Mat& m = this->dfdx[i];
            for( int j=0; j<N; ++j )
                for( int k=0; k<N; ++k )
                    m[j][k] = 0.0;
        }
    }
}

template <class DataTypes>
SReal RepulsiveSpringForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    msg_error() << "RepulsiveSpringForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}

} //namespace sofa::component::interactionforcefield
