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
#include <sofa/component/solidmechanics/spring/RepulsiveSpringForceField.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.inl>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::solidmechanics::spring
{



template<class DataTypes>
void RepulsiveSpringForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{

    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    const type::vector<Spring>& springs= this->d_springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    for (unsigned int i=0; i<springs.size(); i++)
    {
        int a = springs[i].m1;
        int b = springs[i].m2;
        Coord u = x2[b]-x1[a];
        Real d = u.norm();
        if (d!=0 && d < springs[i].initpos)
        {
            Real inverseLength = 1.0f/d;
            u *= inverseLength;
            Real elongation = (Real)(d - springs[i].initpos);
            Deriv relativeVelocity = v2[b]-v1[a];
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = (Real)(springs[i].ks*elongation+springs[i].kd*elongationVelocity);
            Deriv force = u*forceIntensity;
            f1[a]+=force;
            f2[b]-=force;

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

    data_f1.endEdit();
    data_f2.endEdit();
}

template <class DataTypes>
SReal RepulsiveSpringForceField<DataTypes>::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const
{
    msg_error() << "RepulsiveSpringForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}

} //namespace sofa::component::solidmechanics::spring
