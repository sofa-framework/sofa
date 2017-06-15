/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef FRAME_FRAMESPRINGFORCEFIELD2_INL
#define FRAME_FRAMESPRINGFORCEFIELD2_INL

#include "FrameSpringForceField2.h"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
FrameSpringForceField2<DataTypes>::FrameSpringForceField2(MState* obj)
    : Inherit( obj), maskInUse(false)
    , youngModulus ( initData ( &youngModulus, 2000.0, "youngModulus","Young Modulus" ) )
    , poissonRatio ( initData ( &poissonRatio, 0.3, "poissonRatio","Poisson Ratio." ) )
{
}

template<class DataTypes>
FrameSpringForceField2<DataTypes>::FrameSpringForceField2()
    : maskInUse(false)
    , youngModulus ( initData ( &youngModulus, 2000.0, "youngModulus","Young Modulus" ) )
    , poissonRatio ( initData ( &poissonRatio, 0.3, "poissonRatio","Poisson Ratio." ) )
{
}


template <class DataTypes>
void FrameSpringForceField2<DataTypes>::reinit()
{
    getH_isotropic ( H, youngModulus.getValue(), poissonRatio.getValue() );
    computeK0();
}


template <class DataTypes>
void FrameSpringForceField2<DataTypes>::init()
{
    dqInfos = NULL;
    this->Inherit::init();
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::bwdInit()
{
    // Get the first DQStorage which has 'computeAllMatrices' to true
    dqInfos = NULL;
    vector<FStorage*> vFStorage;
    sofa::core::objectmodel::BaseContext* context=  this->getContext();
    context->get<FStorage>( &vFStorage, core::objectmodel::BaseContext::SearchDown);
    FStorage* tmpDqStorage = NULL;
    for( typename vector<FStorage *>::iterator it = vFStorage.begin(); it != vFStorage.end(); it++)
    {
        tmpDqStorage = (*it);
        if( tmpDqStorage && tmpDqStorage->computeAllMatrices.getValue() )
        {
            dqInfos = tmpDqStorage;
            break;
        }
    }

    if(! dqInfos)
    {
        serr << "Can not find the skinning mappping component." << sendl;
        return;
    }
    B = & ( dqInfos->B );
    det = & ( dqInfos->det );
    ddet = & ( dqInfos->ddet );
    vol = & ( dqInfos->vol );

    getH_isotropic ( H, youngModulus.getValue(), poissonRatio.getValue() );
    computeK0();
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::computeK0()
{
    // Compute K
    unsigned int size = this->getMState()->getX()->size();
    K0.resize ( size );
    for ( unsigned int i = 0; i < size; i++ )
    {
        K0[i].resize ( size );
        for ( unsigned int j = 0; j < size; j++ )
            K0[i][j].fill ( 0 );
    }

    // K=-B^T.H.B
    const int nbP=(*B).size();
    if ( nbP==0 ) return;
    const int nbDOF=(*B)[0].size();
    int i,j,k,l,m;
    Mat6xIn HB;
    MatInx6 BT;
    MatInxIn BTHB;
    for ( i=0; i<nbP; ++i )
        for ( j=0; j<nbDOF; ++j )
        {
            HB= ( Real ) ( (*vol)[i] ) *H*(*B)[i][j];
            for ( k=0; k<nbDOF; ++k )
            {
                BT.transpose ( (*B)[i][k] );
                BTHB=BT*HB;
                K0[k][j]-=BTHB;
            }
        }


    // update KSpring (K expressed at the joint location)
    helper::ReadAccessor<VecCoord> xiref = *this->getMState()->getX0();
    Mat33 crossi,crossj;
    Mat66 Kbackup;
    Vec3 tpref;

    for(i=0; i<nbDOF; i++)
        for(j=0; j<nbDOF; j++)
            if(i>j)
            {
                Kbackup=K0[i][j];
                tpref=(xiref[j].getCenter() + xiref[i].getCenter())/2.;		// pivot = center

                GetCrossproductMatrix(crossi,tpref-xiref[i].getCenter());            GetCrossproductMatrix(crossj,tpref-xiref[j].getCenter());

                for(l=0; l<3; l++)  for(m=0; m<3; m++) for(k=0; k<3; k++)
                        {
                            K0[i][j][l][m+3] -= crossi[l][k]*K0[i][j][k+3][m+3];
                            K0[i][j][l+3][m] += K0[i][j][l+3][k+3]*crossj[k][m];
                        }
                for(l=0; l<3; l++)  for(m=0; m<3; m++) for(k=0; k<3; k++)
                            K0[i][j][l][m] += K0[i][j][l][k+3]*crossj[k][m] - crossi[l][k]*Kbackup[k+3][m];

                K0[j][i].transpose(K0[i][j]);
            }

    /*
    for ( i=0;i<nbDOF;++i )
    	for ( j=0;j<nbDOF;++j )
    		if(i!=j)
    			{
    			for(l=0;l<6;l++)  for(m=0;m<6;m++) if(fabs(K0[i][j][l][m])<1E-5) K0[i][j][l][m]=0;
    			serr << "Kspring["<<i<<"]["<<j<<"]: " << K0[i][j] << sendl;
    			}
    	 */

}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::addForce(VecDeriv& vf, const VecCoord& vx, const VecDeriv& /*vv*/)
{
    if( vf.size() != K0.size())
    {
        getH_isotropic ( H, youngModulus.getValue(), poissonRatio.getValue() );
        computeK0();
    }


    vf.resize(vx.size());

    const unsigned int size=vx.size();
    // Compute K
    K.resize ( size );
    for ( unsigned int i = 0; i < size; ++i )
    {
        K[i].resize ( size );
        for ( unsigned int j = 0; j < size; j++ )
            K[i][j].fill ( 0 );
    }

    updateForce( vf, K, vx, K0 );
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx )
{
    df.resize ( dx.size() );

    for ( unsigned int i = 0; i < dx.size(); ++i )
    {
        Vec6 res = Vec6 ( 0, 0, 0, 0, 0, 0 );
        for ( unsigned int j = 0; j < dx.size(); ++j )
        {
//                        Vec6 tmp;
//                        tmp[0] = dx[j].getVOrientation() [0];
//                        tmp[1] = dx[j].getVOrientation() [1];
//                        tmp[2] = dx[j].getVOrientation() [2];
//                        tmp[3] = dx[j].getVCenter() [0];
//                        tmp[4] = dx[j].getVCenter() [1];
//                        tmp[5] = dx[j].getVCenter() [2];
//                        res += K[i][j] * tmp;
            df[i] += K[i][j] * dx[j];
        }
//                    df[i].getVOrientation() [0] += res[0];
//                    df[i].getVOrientation() [1] += res[1];
//                    df[i].getVOrientation() [2] += res[2];
//                    df[i].getVCenter() [0] += res[3];
//                    df[i].getVCenter() [1] += res[4];
//                    df[i].getVCenter() [2] += res[5];
    }
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::draw()
{

}



template<class DataTypes>
void FrameSpringForceField2<DataTypes>::updateForce( VecDeriv& Force, VVMatInxIn& K, const VecCoord& xi, const VVMatInxIn& Kref )
{
    // generalized spring joint network based on precomputed FrameHooke stiffness matrices
    helper::ReadAccessor<VecCoord> xiref = *this->getMState()->getX0();
    int i,j,k,l,m,nbDOF=xi.size();
    double n;
    Coord MI,MJ,M;
    Deriv Thetaij,Fm;
    Vec3 mbar;
    Quat q;
    Mat33 crossli,crosslj;
    Mat66 Km,Ktemp,KmSym;
    Mat33 Rm;

    for (i=0; i<nbDOF; i++)
    {
        for (j=0; j<nbDOF; j++)
            if (i>j)
            {
                mbar=(xiref[j].getCenter() + xiref[i].getCenter())/2.;    // pivot = center

                q=xiref[i].getOrientation();
                q[3]*=-1;
                Multi_Q(MI.getOrientation(),xi[i].getOrientation(),q);
                Transform_Q(MI.getCenter(),mbar-xiref[i].getCenter(),MI.getOrientation());
                MI.getCenter()+=xi[i].getCenter();
                q=xiref[j].getOrientation();
                q[3]*=-1;
                Multi_Q(MJ.getOrientation(),xi[j].getOrientation(),q);
                Transform_Q(MJ.getCenter(),mbar-xiref[j].getCenter(),MJ.getOrientation());
                MJ.getCenter()+=xi[j].getCenter();

                // in ref pos: Theta.dt=MJ.MI-1
                PostoSpeed(Thetaij,MI,MJ);

                // mid frame M
                M.getCenter()=(MI.getCenter()+MJ.getCenter())/2.;
                n=getVOrientation(Thetaij)*getVOrientation(Thetaij);
                if (n>0)
                {
                    n=sqrt(n);
                    q[3]=cos(n/4.);
                    n=sin(n/4.)/n;
                    q[0]=getVOrientation(Thetaij)[0]*n;
                    q[1]=getVOrientation(Thetaij)[1]*n;
                    q[2]=getVOrientation(Thetaij)[2]*n;
                }
                else
                {
                    q[3]=1;
                    q[0]=q[1]=q[2]=0;
                }

                Multi_Q(M.getOrientation(),q,MI.getOrientation());
                QtoR(Rm,M.getOrientation());

                // rotate Kspring (=Kmbar in the paper) in deformed space
                Km=Kref[j][i];

                Ktemp.fill(0);
                for (k=0; k<3; k++)  for (l=0; l<3; l++) for (m=0; m<3; m++)
                        {
                            Ktemp[k][l]+=Rm[k][m]*Km[m][l];
                            Ktemp[k][l+3]+=Rm[k][m]*Km[m][l+3];
                            Ktemp[k+3][l]+=Rm[k][m]*Km[m+3][l];
                            Ktemp[k+3][l+3]+=Rm[k][m]*Km[m+3][l+3];
                        }
                Km.fill(0);
                for (k=0; k<3; k++)  for (l=0; l<3; l++) for (m=0; m<3; m++)
                        {
                            Km[k][l]+=Ktemp[k][m]*Rm[l][m];
                            Km[k][l+3]+=Ktemp[k][m+3]*Rm[l][m];
                            Km[k+3][l]+=Ktemp[k+3][m]*Rm[l][m];
                            Km[k+3][l+3]+=Ktemp[k+3][m+3]*Rm[l][m];
                        }
                KmSym=(Km+Km.transposed())/2.;

                // spring force at M
//                        Fm=Deriv();
//                        for (k=0;k<3;k++) for (l=0;l<3;l++)
//                        {
//                            Fm.getVOrientation()[k]+=KmSym[k][l]*Thetaij.getVOrientation()[l];
//                            Fm.getVOrientation()[k]+=KmSym[k][l+3]*Thetaij.getVCenter()[l];
//                            Fm.getVCenter()[k]+=KmSym[k+3][l]*Thetaij.getVOrientation()[l];
//                            Fm.getVCenter()[k]+=KmSym[k+3][l+3]*Thetaij.getVCenter()[l];
//                        }
                Fm=KmSym*Thetaij;

                // force on i (displaced from M)
                getVOrientation(Force[i])+=getVOrientation(Fm)+cross(M.getCenter()-xi[i].getCenter(),getVCenter(Fm));
                getVCenter(Force[i])+=getVCenter(Fm);

                // force on j (displaced from M)
                getVOrientation(Force[j]) -= getVOrientation(Fm) + cross(M.getCenter()-xi[j].getCenter(),getVCenter(Fm));
                getVCenter(Force[j]) -= getVCenter(Fm);

                // update K
                if (K.size())
                {
                    GetCrossproductMatrix(crossli,M.getCenter()-xi[i].getCenter());
                    GetCrossproductMatrix(crosslj,M.getCenter()-xi[j].getCenter());

                    Ktemp=Km;
                    for (l=0; l<3; l++)  for (m=0; m<3; m++) for (k=0; k<3; k++)
                            {
                                Ktemp[l][m]  -= Ktemp[l][k+3]  * crossli[k][m];
                                Ktemp[l+3][m]-= Ktemp[l+3][k+3]* crossli[k][m];
                            }
                    K[j][i]=Ktemp;
                    K[i][i]-=Ktemp;
                    for (l=0; l<3; l++)  for (m=0; m<3; m++) for (k=0; k<3; k++)
                            {
                                K[j][i][l][m]   += crosslj[l][k]*Ktemp[k+3][m];
                                K[j][i][l][m+3] += crosslj[l][k]*Ktemp[k+3][m+3];
                                K[i][i][l][m]   -= crossli[l][k]*Ktemp[k+3][m];
                                K[i][i][l][m+3] -= crossli[l][k]*Ktemp[k+3][m+3];
                            }
                    K[i][j]=K[j][i].transposed();

                    Ktemp=Km.transposed();
                    for (l=0; l<3; l++)  for (m=0; m<3; m++) for (k=0; k<3; k++)
                            {
                                Ktemp[l][m]  -= Ktemp[l][k+3]  * crosslj[k][m];
                                Ktemp[l+3][m]-= Ktemp[l+3][k+3]* crosslj[k][m];
                            }
                    K[j][j]-=Ktemp;
                    for (l=0; l<3; l++)  for (m=0; m<3; m++) for (k=0; k<3; k++)
                            {
                                K[j][j][l][m]   -= crosslj[l][k]*Ktemp[k+3][m];
                                K[j][j][l][m+3] -= crosslj[l][k]*Ktemp[k+3][m+3];
                            }
                }

            }

    }
    /*
    // test momentum conservation
    Fm=Deriv();
    for(i=0;i<nbDOF;i++)
    {
    Fm.getVOrientation()+=Force[i].getVOrientation()+cross(Force[i].getVCenter(),-xi[i].getCenter());
    Fm.getVCenter()+=Force[i].getVCenter();
    }
    cerr<<"w:"<<Fm.getVOrientation().norm()<<"v:"<<Fm.getVCenter().norm();*/
}




template<class DataTypes>
void FrameSpringForceField2<DataTypes>::QtoR( Mat33& M, const Quat& q)
{
    // q to M
    double xs = q[0]*2., ys = q[1]*2., zs = q[2]*2.;
    double wx = q[3]*xs, wy = q[3]*ys, wz = q[3]*zs;
    double xx = q[0]*xs, xy = q[0]*ys, xz = q[0]*zs;
    double yy = q[1]*ys, yz = q[1]*zs, zz = q[2]*zs;
    M[0][0] = 1.0 - (yy + zz); M[0][1]= xy - wz; M[0][2] = xz + wy;
    M[1][0] = xy + wz; M[1][1] = 1.0 - (xx + zz); M[1][2] = yz - wx;
    M[2][0] = xz - wy; M[2][1] = yz + wx; M[2][2] = 1.0 - (xx + yy);
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::Transform_Q( Vec3& pout, const Vec3& pin, const Quat& q, const bool& invert)
{
    double as = q[0]*2., bs = q[1]*2., cs = q[2]*2.;
    double wa = q[3]*as, wb = q[3]*bs, wc = q[3]*cs;
    double aa = q[0]*as, ab = q[0]*bs, ac = q[0]*cs;
    double bb = q[1]*bs, bc = q[1]*cs, cc = q[2]*cs;

    if(invert) {wa*=-1; wb*=-1; wc*=-1;}

    pout[0]= pin[0]* (1.0 - (bb + cc)) + pin[1]* (ab - wc) + pin[2]* (ac + wb);
    pout[1]= pin[0]* (ab + wc) + pin[1]* (1.0 - (aa + cc)) + pin[2]* ( bc - wa);
    pout[2]= pin[0]* (ac - wb) + pin[1]* (bc + wa) + pin[2]* (1.0 - (aa + bb));
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::PostoSpeed( Deriv& Omega, const Coord& xi, const Coord& xi2)
{
    // X2=Omega*X -> Q(Omega/2)=X2*X^-1

    Quat q_inv; q_inv[0]=-xi.getOrientation()[0];    q_inv[1]=-xi.getOrientation()[1];    q_inv[2]=-xi.getOrientation()[2];
    q_inv[3]=xi.getOrientation()[3];
    Quat Om; Multi_Q( Om,xi2.getOrientation(),q_inv);

    double W=0,n=sqrt(Om[0]*Om[0]+Om[1]*Om[1]+Om[2]*Om[2]);
    if(Om[3]<-1) Om[3]=-1;
    if(n!=0 && Om[3]<=1 && Om[3]>=-1) W=2*acos(Om[3])/n;
    getVOrientation(Omega)[0]=Om[0]*W;
    getVOrientation(Omega)[1]=Om[1]*W;
    getVOrientation(Omega)[2]=Om[2]*W;
    getVCenter(Omega)=xi2.getCenter()-xi.getCenter();
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::Multi_Q( Quat& q, const Quat& q1, const Quat& q2)
{
    Vec3 qv1; qv1[0]=q1[0]; qv1[1]=q1[1]; qv1[2]=q1[2];
    Vec3 qv2; qv2[0]=q2[0]; qv2[1]=q2[1]; qv2[2]=q2[2];
    q[3]=q1[3]*q2[3]-qv1*qv2;
    Vec3 cr=cross(qv1,qv2);  q[0]=cr[0]; q[1]=cr[1]; q[2]=cr[2];
    q[0]+=q2[3]*q1[0]+q1[3]*q2[0];
    q[1]+=q2[3]*q1[1]+q1[3]*q2[1];
    q[2]+=q2[3]*q1[2]+q1[3]*q2[2];
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::getH_isotropic ( Mat66& H, const double& E, const double& v )
{
    Real c=(Real)(E/ ( ( 1+v ) * ( 1-2*v ) ));
    for ( int i=0; i<6; i++ ) for ( int j=0; j<6; j++ ) H[i][j]=0;
    H[0][0]=c* ( 1-v );
    H[1][1]=c* ( 1-v );
    H[2][2]=c* ( 1-v );
    H[3][3]=c* ( 1-2*v );
    H[4][4]=c* ( 1-2*v );
    H[5][5]=c* ( 1-2*v );
    H[1][0]=H[0][1]=c*v;
    H[2][0]=H[0][2]=c*v;
    H[2][1]=H[1][2]=c*v;
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::GetCrossproductMatrix(Mat33& C,const Vec3& u)
{
    C[0][0]=0;	   C[0][1]=-u[2];  C[0][2]=u[1];
    C[1][0]=u[2];   C[1][1]=0;      C[1][2]=-u[0];
    C[2][0]=-u[1];  C[2][1]=u[0];   C[2][2]=0;
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
