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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/forcefield/SpringForceField.inl>
#include <assert.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void StiffSpringForceField<DataTypes>::init()
{
    this->SpringForceField<DataTypes>::init();
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringForce( double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    if( d>1.0e-4 )
    {
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)(d - spring.initpos);
        potentialEnergy += elongation * elongation * spring.ks / 2;
        /*                    serr<<"StiffSpringForceField<DataTypes>::addSpringForce, p1 = "<<p1<<sendl;
                            serr<<"StiffSpringForceField<DataTypes>::addSpringForce, p2 = "<<p2<<sendl;
                            serr<<"StiffSpringForceField<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<sendl;*/
        Deriv relativeVelocity = v2[b]-v1[a];
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
        Deriv force = u*forceIntensity;
        f1[a]+=force;
        f2[b]-=force;
        if (this->maskInUse)
        {
            this->mstate1->forceMask.insertEntry(a);
            this->mstate2->forceMask.insertEntry(b);
        }
        Mat& m = this->dfdx[i];
        Real tgt = forceIntensity * inverseLength;
        for( int j=0; j<N; ++j )
        {
            for( int k=0; k<N; ++k )
            {
                m[j][k] = ((Real)spring.ks-tgt) * u[j] * u[k];
            }
            m[j][j] += tgt;
        }
    }
    else // null length, no force and no stiffness
    {
        Mat& m = this->dfdx[i];
        for( int j=0; j<N; ++j )
        {
            for( int k=0; k<N; ++k )
            {
                m[j][k] = 0;
            }
        }
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int i, const Spring& spring, double kFactor, double /*bFactor*/)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Coord d = dx2[b]-dx1[a];
    Deriv dforce = this->dfdx[i]*d;
    dforce *= kFactor;
    f1[a]+=dforce;
    f2[b]-=dforce;
    //serr<<"StiffSpringForceField<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<sendl;
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    const helper::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    //serr<<"StiffSpringForceField<DataTypes>::addForce()"<<sendl;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        //serr<<"StiffSpringForceField<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<sendl;
        this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor)
{
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, dx1 = "<<dx1<<sendl;
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df1 before = "<<f1<<sendl;
    const helper::vector<Spring>& springs = this->springs.getValue();
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1,dx1,df2,dx2, i, springs[i], kFactor, bFactor);
    }
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df1 = "<<f1<<sendl;
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df2 = "<<f2<<sendl;
}




template<class DataTypes>
void StiffSpringForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, double kFact, unsigned int &offset)
{
    const sofa::helper::vector<Spring >& ss = this->springs.getValue();

    for (unsigned int e=0; e<ss.size(); e++)
    {
        const Spring& s = ss[e];

        unsigned p1 = offset+N*s.m1;
        unsigned p2 = offset+N*s.m2;

        for(int i=0; i<N; i++)
            for (int j=0; j<N; j++)
            {
                Real k = (Real)(this->dfdx[e][i][j]*kFact);

                mat->add(p1+i,p1+j, -k);
                mat->add(p1+i,p2+j, k);
                mat->add(p2+i,p1+j, k);//or mat->add(p1+j,p2+i, k);
                mat->add(p2+i,p2+j, -k);
            }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif

