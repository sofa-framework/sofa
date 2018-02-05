/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_INL

#include <SofaDeformable/StiffSpringForceField.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{


template<class DataTypes>
void StiffSpringForceField<DataTypes>::init()
{
    this->SpringForceField<DataTypes>::init();
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringForce(
        Real& potentialEnergy,
        VecDeriv& f1,
        const  VecCoord& p1,
        const VecDeriv& v1,
        VecDeriv& f2,
        const  VecCoord& p2,
        const  VecDeriv& v2,
        int i,
        const Spring& spring)
{
    //    this->cpt_addForce++;
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    if( spring.enabled && d>1.0e-9 && (!spring.elongationOnly || d>spring.initpos))
    {
        // F =   k_s.(l-l_0 ).U + k_d((V_b - V_a).U).U = f.U   where f is the intensity and U the direction
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)(d - spring.initpos);
        potentialEnergy += elongation * elongation * spring.ks / 2;
        Deriv relativeVelocity = v2[b]-v1[a];
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
        Deriv force = u*forceIntensity;

        f1[a]+=force;
        f2[b]-=force;

        // Compute stiffness dF/dX
        // The force change dF comes from length change dl and unit vector change dU:
        // dF = k_s.dl.U + f.dU
        // dU = 1/l.(I-U.U^T).dX   where dX = dX_1 - dX_0  and I is the identity matrix
        // dl = U^T.dX
        // dF = k_s.U.U^T.dX + f/l.(I-U.U^T).dX = ((k_s-f/l).U.U^T + f/l.I).dX
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
void StiffSpringForceField<DataTypes>::addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, int i, const Spring& spring, double kFactor, double /*bFactor*/)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Coord d = dx2[b]-dx1[a];
    Deriv dforce = this->dfdx[i]*d;
    dforce *= kFactor;

    df1[a]+=dforce;
    df2[b]-=dforce;
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{
    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    const helper::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    Real kFactor       =  (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    Real bFactor       =  (Real)mparams->bFactor();

    const helper::vector<Spring>& springs = this->springs.getValue();
    df1.resize(dx1.size());
    df2.resize(dx2.size());

    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1,dx1,df2,dx2, i, springs[i], kFactor, bFactor);
    }

    data_df1.endEdit();
    data_df2.endEdit();
}




template<class DataTypes>
void StiffSpringForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{

    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
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
            {
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
            {
                for(int i=0; i<N; i++)
                {
                    for (int j=0; j<N; j++)
                    {
                        mat11.matrix->add(mat11.offset+p1+i,mat11.offset+p1+j, -(Real)m[i][j]);
                    }
                }
            }
            if (mat12)
            {
                for(int i=0; i<N; i++)
                {
                    for (int j=0; j<N; j++)
                    {
                        mat12.matrix->add(mat12.offRow+p1+i,mat12.offCol+p2+j,  (Real)m[i][j]);
                    }
                }
            }
            if (mat21)
            {
                for(int i=0; i<N; i++)
                {
                    for (int j=0; j<N; j++)
                    {
                        mat21.matrix->add(mat21.offRow+p2+i,mat21.offCol+p1+j,  (Real)m[i][j]);
                    }
                }
            }
            if (mat22)
            {
                for(int i=0; i<N; i++)
                {
                    for (int j=0; j<N; j++)
                    {
                        mat22.matrix->add(mat22.offset+p2+i,mat22.offset+p2+j, -(Real)m[i][j]);
                    }
                }
            }
        }
    }

}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_INL */

