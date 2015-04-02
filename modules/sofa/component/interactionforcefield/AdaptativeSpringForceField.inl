/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
// Author: Rosalie Plantefeve, INRIA, (C) 2013

// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_AdaptativeSpringForceField_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_AdaptativeSpringForceField_INL

#include <sofa/component/interactionforcefield/AdaptativeSpringForceField.h>
#include <SofaDeformable/SpringForceField.inl>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
void AdaptativeSpringForceField<DataTypes>::init()
{
    this->SpringForceField<DataTypes>::init();
}

template<class DataTypes>
void AdaptativeSpringForceField<DataTypes>::addSpringForce(
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
    if( d>1.0e-4 )
    {
        // F =   k_s.(l-l_0 ).U + k_d((V_b - V_a).U).U = f.U   where f is the intensity and U the direction
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)(d - spring.initpos);
        potentialEnergy += elongation * elongation * spring.ks / 2;
        //                    serr<<"AdaptativeSpringForceField<DataTypes>::addSpringForce, p1 = "<<p1<<sendl;
        //                    serr<<"AdaptativeSpringForceField<DataTypes>::addSpringForce, p2 = "<<p2<<sendl;
        //                    serr<<"AdaptativeSpringForceField<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<sendl;
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
void AdaptativeSpringForceField<DataTypes>::addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, int i, const Spring& spring, double kFactor, double /*bFactor*/)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Coord d = dx2[b]-dx1[a];
    Deriv dforce = this->dfdx[i]*d;
    dforce *= kFactor;
    //                serr<<"AdaptativeSpringForceField<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dx1 ="<<  dx1 <<", dx2 ="<<  dx2 <<sendl;
    //                serr<<"AdaptativeSpringForceField<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<sendl;
    df1[a]+=dforce;
    df2[b]-=dforce;
    //                serr<<"AdaptativeSpringForceField<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", df1 after ="<<df1<<", df2 after ="<<df2<<sendl;
}

template<class DataTypes>
void AdaptativeSpringForceField<DataTypes>::addForce(
        const MechanicalParams* /*mparams*/ /* PARAMS FIRST */,
        DataVecDeriv& data_f1,
        DataVecDeriv& data_f2,
        const DataVecCoord& data_x1,
        const DataVecCoord& data_x2,
        const DataVecDeriv& data_v1,
        const DataVecDeriv& data_v2 )
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
void AdaptativeSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    double kFactor       =  mparams->kFactor();
    double bFactor       =  mparams->bFactor();

    const helper::vector<Spring>& springs = this->springs.getValue();
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    //serr<<"AdaptativeSpringForceField<DataTypes>::addDForce, dx1 = "<<dx1<<sendl;
    //serr<<"AdaptativeSpringForceField<DataTypes>::addDForce, df1 before = "<<f1<<sendl;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1,dx1,df2,dx2, i, springs[i], kFactor, bFactor);
    }
    //serr<<"AdaptativeSpringForceField<DataTypes>::addDForce, df1 = "<<f1<<sendl;
    //serr<<"AdaptativeSpringForceField<DataTypes>::addDForce, df2 = "<<f2<<sendl;

    data_df1.endEdit();
    data_df2.endEdit();
}




template<class DataTypes>
void AdaptativeSpringForceField<DataTypes>::addKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{

    double kFact = mparams->kFactor();
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
                        mat21.matrix->add(mat21.offRow+p1+i,mat21.offCol+p2+j,  (Real)m[i][j]);
                    }
                }
            }
            if (mat22)
            {
                for(int i=0; i<N; i++)
                {
                    for (int j=0; j<N; j++)
                    {
                        mat22.matrix->add(mat22.offset+p2+i,mat11.offset+p2+j, -(Real)m[i][j]);
                    }
                }
            }
        }
    }

}
template<class DataTypes>
void AdaptativeSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();


    std::vector< Vector3 > points;

    const helper::vector<Spring>& springs = this->springs.getValue();

    Real ksMean=0;
    Real ksSigma=0;
    Real ksMax=0;
    Real ksMin=500000000;

    for (unsigned int i=0; i<springs.size(); i++)
    {
        if (springs[i].ks<ksMin)
        {
            ksMin=springs[i].ks;
        }
        if (springs[i].ks>ksMax)
        {
            ksMax=springs[i].ks;
        }
        ksMean+=springs[i].ks;
    }
    ksMean/=springs.size();

    for (unsigned int i=0; i<springs.size(); i++)
    {
        ksSigma+=(springs[i].ks-ksMean)*(springs[i].ks-ksMean);
    }
    ksSigma/=springs.size();
    ksSigma=sqrt(ksSigma);

    Real sepNb=100.0;
    Real d = 1/sepNb;

    //std::cout << "ksMin = " << ksMin << " ksMax = " << ksMax << " d = " << d <<std::endl;
    const Vec<4,float> c0(1,0.5,0,1);
    Vec<4,float> c2(1,1,1,1);

    for (unsigned int i=0; i<springs.size(); i++)
    {
        Real stiffness = springs[i].ks;
        //std::cout << "stiffness = " << stiffness <<std::endl;
        Vector3 point2,point1;
        point1 = DataTypes::getCPos(p1[springs[i].m1]);
        point2 = DataTypes::getCPos(p2[springs[i].m2]);

        points.push_back(point1);
        points.push_back(point2);



        for (unsigned int j=0; j<sepNb; j++)
        {
            Real k= (Real) j;

            if((ksMean-ksSigma*(j+1)*d)<=stiffness && stiffness<=(ksMean-ksSigma*(j)*d))
            {
                c2[0] = (float)( 0.5-0.5*k/(sepNb) );
                c2[1] = (float)( 0.5+0.5*(k/(sepNb)) );
                c2[2] = (float)( 0.25+0.25*(k/(sepNb)) );
            }
            else if((ksMean+ksSigma*j*d)<=stiffness && stiffness<=(ksMean+ksSigma*(j+1)*d))
            {
                c2[0] = (float)( 0.5+0.5*k/(sepNb) );
                c2[1] = (float)( 0.5-0.5*(k/(sepNb)) );
                c2[2] = (float)( 0.25-0.25*(k/(sepNb)) );
            }
            else if (stiffness<(ksMean-ksSigma*d*sepNb))
            {
                c2[0] = 0;
                c2[1] = 1;
                c2[2] = 0.5f;
            }
            else if (stiffness>(ksMean+ksSigma*d*sepNb))
            {
                c2[0] = 1;
                c2[1] = 0;
                c2[2] = 0;
            }



        }

        //std::cout << "draw i=" << i << " k = " << valeurK << " " << c2[0] << " " << c2[1] << " " << c2[2] <<std::endl;
        vparams->drawTool()->drawPoints(points,5, c2);
        points.pop_back();
        points.pop_back();

    }








}
} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_AdaptativeSpringForceField_INL */

