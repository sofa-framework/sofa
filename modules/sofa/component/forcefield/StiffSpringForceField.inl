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
void StiffSpringForceField<DataTypes>::addSpringForce( double& potentialEnergy, WRefVecDeriv& f1, RRefVecCoord& p1, RRefVecDeriv& v1, WRefVecDeriv& f2, RRefVecCoord& p2, RRefVecDeriv& v2, int i, const Spring& spring)
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
void StiffSpringForceField<DataTypes>::addSpringDForce(WRefVecDeriv& f1, RRefVecDeriv& dx1, WRefVecDeriv& f2, RRefVecDeriv& dx2, int i, const Spring& spring, double kFactor, double /*bFactor*/)
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
void StiffSpringForceField<DataTypes>::addForce(VecDeriv& vf1, VecDeriv& vf2, const VecCoord& vx1, const VecCoord& vx2, const VecDeriv& vv1, const VecDeriv& vv2)
{

    WRefVecDeriv f1 = vf1;
    RRefVecCoord x1 = vx1;
    RRefVecDeriv v1 = vv1;
    WRefVecDeriv f2 = vf2;
    RRefVecCoord x2 = vx2;
    RRefVecDeriv v2 = vv2;

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
void StiffSpringForceField<DataTypes>::addDForce(VecDeriv& vdf1, VecDeriv& vdf2, const VecDeriv& vdx1, const VecDeriv& vdx2, double kFactor, double bFactor)
{
    WRefVecDeriv df1 = vdf1;
    WRefVecDeriv df2 = vdf2;
    RRefVecDeriv dx1 = vdx1;
    RRefVecDeriv dx2 = vdx2;
    const helper::vector<Spring>& springs = this->springs.getValue();
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, dx1 = "<<dx1<<sendl;
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df1 before = "<<f1<<sendl;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1,dx1,df2,dx2, i, springs[i], kFactor, bFactor);
    }
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df1 = "<<f1<<sendl;
    //serr<<"StiffSpringForceField<DataTypes>::addDForce, df2 = "<<f2<<sendl;
}




template<class DataTypes>
void StiffSpringForceField<DataTypes>::addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double kFact)
{
    if (this->mstate1 == this->mstate2)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat = matrix->getMatrix(this->mstate1);
        if (!mat) return;
        const sofa::helper::vector<Spring >& ss = this->springs.getValue();
        const unsigned int n = ss.size() < this->dfdx.size() ? ss.size() : this->dfdx.size();
        for (unsigned int e=0; e<n; e++)
        {
            const Spring& s = ss[e];
            unsigned p1 = mat.offset+Deriv::total_size*s.m1;
            unsigned p2 = mat.offset+Deriv::total_size*s.m2;
            const Mat& m = this->dfdx[e];
            for(int i=0; i<N; i++)
                for (int j=0; j<N; j++)
                {
                    Real k = (Real)(m[i][j]*kFact);
                    mat.matrix->add(p1+i,p1+j, -k);
                    mat.matrix->add(p1+i,p2+j, k);
                    mat.matrix->add(p2+i,p1+j, k);//or mat->add(p1+j,p2+i, k);
                    mat.matrix->add(p2+i,p2+j, -k);
                }
        }
    }
    else
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(this->mstate1);
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(this->mstate2);
        sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(this->mstate1, this->mstate2);
        sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(this->mstate2, this->mstate1);
        if (!mat11 && !mat22 && !mat12 && !mat21) return;
        const sofa::helper::vector<Spring >& ss = this->springs.getValue();
        const unsigned int n = ss.size() < this->dfdx.size() ? ss.size() : this->dfdx.size();
        for (unsigned int e=0; e<n; e++)
        {
            const Spring& s = ss[e];
            unsigned p1 = /*mat.offset+*/Deriv::total_size*s.m1;
            unsigned p2 = /*mat.offset+*/Deriv::total_size*s.m2;
            Mat m = this->dfdx[e]* (Real) kFact;
            if (mat11)
                for(int i=0; i<N; i++) for (int j=0; j<N; j++) mat11.matrix->add(mat11.offset+p1+i,mat11.offset+p1+j, -(Real)m[i][j]);
            if (mat12)
                for(int i=0; i<N; i++) for (int j=0; j<N; j++) mat12.matrix->add(mat12.offRow+p1+i,mat12.offCol+p1+j,  (Real)m[i][j]);
            if (mat21)
                for(int i=0; i<N; i++) for (int j=0; j<N; j++) mat21.matrix->add(mat21.offRow+p1+i,mat21.offCol+p1+j,  (Real)m[i][j]);
            if (mat22)
                for(int i=0; i<N; i++) for (int j=0; j<N; j++) mat22.matrix->add(mat22.offset+p2+i,mat11.offset+p2+j, -(Real)m[i][j]);
        }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif

