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
#include <sofa/component/solidmechanics/spring/JointSpring.h>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
JointSpring<DataTypes>::JointSpring(sofa::Index m1 , sofa::Index m2,
                                    Real softKst, Real hardKst , Real softKsr , Real hardKsr , Real blocKsr,
                                    Real axmin , Real axmax , Real aymin , Real aymax , Real azmin , Real azmax,
                                    Real kd):
                                      m1(m1), m2(m2), kd(kd)
                                    , torsion(0,0,0), lawfulTorsion(0,0,0), KT(0,0,0) , KR(0,0,0)
                                    , softStiffnessTrans(softKst), hardStiffnessTrans(hardKst), softStiffnessRot(softKsr), hardStiffnessRot(hardKsr), blocStiffnessRot(blocKsr)
                                    , needToInitializeTrans(true), needToInitializeRot(true)
{
    limitAngles = sofa::type::Vec<6,Real>(axmin,axmax,aymin,aymax,azmin,azmax);
    freeMovements = sofa::type::Vec<6,bool>(false, false, false, true, true, true);
    for (unsigned int i=0; i<3; i++)
    {
        if(limitAngles[2*i]==limitAngles[2*i+1])
            freeMovements[3+i] = false;
    }
    initTrans = Vector(0,0,0);
    initRot = type::Quat<SReal>(0,0,0,1);
}

} // namespace sofa::component::solidmechanics::spring
