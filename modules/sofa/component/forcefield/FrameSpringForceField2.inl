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
#ifndef SOFA_COMPONENT_FORCEFIELD_FRAMESPRINGFORCEFIELD2_INL
#define SOFA_COMPONENT_FORCEFIELD_FRAMESPRINGFORCEFIELD2_INL

#include <sofa/component/forcefield/FrameSpringForceField2.h>
#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/mapping/SkinningMapping.inl>
#include <sofa/simulation/common/Simulation.h>
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
FrameSpringForceField2<DataTypes>::FrameSpringForceField2(MechanicalState* obj)
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
    sMapping = NULL;
    this->Inherit::init();
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::bwdInit()
{
    this->getContext()->get( sMapping, core::objectmodel::BaseContext::SearchDown);
    if(! sMapping)
    {
        serr << "Can not find the skinning mappping component." << sendl;
        return;
    }
    B = & ( sMapping->B );
    det = & ( sMapping->det );
    ddet = & ( sMapping->ddet );
    vol = & ( sMapping->vol );

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
    const int nbDOF=(*B).size();
    if ( nbDOF==0 ) return;
    int i,j,k;
    const int nbP=(*B)[0].size();
    Mat66 HB,BTHB,BT;
    for ( i=0; i<nbP; ++i )
        for ( j=0; j<nbDOF; ++j )
        {
            HB= ( Real ) ( (*vol)[i] ) *H*(*B)[j][i];
            for ( k=0; k<nbDOF; ++k )
            {
                BT.transpose ( (*B)[k][i] );
                BTHB=BT*HB;
                K0[k][j]-=BTHB;
            }
        }
    /*
    for ( i=0;i<nbDOF;++i )
    for ( j=0;j<nbDOF;++j )
    serr << "K0["<<i<<"]["<<j<<"]: " << K0[i][j] << sendl;
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
            Vec6 tmp;
            tmp[0] = dx[j].getVOrientation() [0];
            tmp[1] = dx[j].getVOrientation() [1];
            tmp[2] = dx[j].getVOrientation() [2];
            tmp[3] = dx[j].getVCenter() [0];
            tmp[4] = dx[j].getVCenter() [1];
            tmp[5] = dx[j].getVCenter() [2];
            res += K[i][j] * tmp;
        }
        df[i].getVOrientation() [0] += res[0];
        df[i].getVOrientation() [1] += res[1];
        df[i].getVOrientation() [2] += res[2];
        df[i].getVCenter() [0] += res[3];
        df[i].getVCenter() [1] += res[4];
        df[i].getVCenter() [2] += res[5];
    }
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::draw()
{

}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::GetIntermediateFrame( Coord& xi, const Coord& x1, const Coord& x2 )
{
// dual quat linear blending with two frames with w=0.5
    int i;
    DUALQUAT bn,b,q1,q2;
    XItoQ( q1, x1);
    XItoQ( q2, x2);
    for(i=0; i<4; i++) {b.q0[i]=0; b.qe[i]=0;}
    for(i=0; i<4; i++) {b.q0[i]+=0.5*q1.q0[i]; b.qe[i]+=0.5*q1.qe[i];}
    for(i=0; i<4; i++) {b.q0[i]+=0.5*q2.q0[i]; b.qe[i]+=0.5*q2.qe[i];}
    double Q0Q0 = b.q0 * b.q0,QEQ0 = b.q0 * b.qe; double Q0= sqrt(Q0Q0);
    for(i=0; i<4; i++) {bn.q0[i]=b.q0[i]/Q0; bn.qe[i]=b.qe[i]/Q0;}
    for(i=0; i<4; i++) bn.qe[i]-=QEQ0*bn.q0[i]/Q0Q0;
    QtoXI(xi,bn);
}


template<class DataTypes>
void FrameSpringForceField2<DataTypes>::updateForce( VecDeriv& Force, VVMat66& K, const VecCoord& xi, const VVMat66& Kref )
{
// force_i =sum Kij Omega_ij.dt + sum (-Kji Omega_ji.dt)_(moved to i)
    VecCoord& xiref = *this->getMState()->getX0();
    int i,j,k,l,m,nbDOF=xi.size();
    Coord Xinv,xjbar,xmean,xmeanref,xmeaninv,T;
    Deriv Theta,df;
    Vec3 OiOj,cr;
    Mat33 R,Crossp;
    Mat66 K2,K3;

    for(i=0; i<nbDOF; i++)
    {
        for(j=0; j<nbDOF; j++)
            if(i>j)
            {
// intermediate frames and registration
                GetIntermediateFrame( xmean,xi[i],xi[j]);
                GetIntermediateFrame( xmeanref,xiref[i],xiref[j]);
                xmeaninv = Coord() - xmean;
                Multi_Rigid( T,xmeanref,xmeaninv);
                QtoR( R,T.getOrientation()); // 3x3 rotation

                OiOj=xi[j].getCenter()-xi[i].getCenter();
                Crossp[0][0]=0; Crossp[0][1]=-OiOj[2]; Crossp[0][2]=OiOj[1];
                Crossp[1][0]=OiOj[2]; Crossp[1][1]=0; Crossp[1][2]=-OiOj[0];
                Crossp[2][0]=-OiOj[1]; Crossp[2][1]=OiOj[0]; Crossp[2][2]=0;

//action of j
                Multi_Rigid( xjbar,T,xi[j]);
// in ref pos: Theta.dt=jbar.jo-1
                PostoSpeed( Theta,xiref[j],xjbar);
// in current pos: Df_i= R(irel_q^1) Kij Theta.dt = K2 Theta.dt
                K2.fill(0);
                for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                        {
                            K2[l][m]+=Kref[i][j][k][m] * R[k][l];
                            K2[l][m+3]+=Kref[i][j][k][m+3] * R[k][l];
                            K2[l+3][m]+=Kref[i][j][k+3][m] * R[k][l];
                            K2[l+3][m+3]+=Kref[i][j][k+3][m+3] * R[k][l];
                        }
                for(k=0; k<3; k++) for(l=0; l<3; l++)
                    {
                        df.getVOrientation()[k]+=K2[k][l]*Theta.getVOrientation()[l]; df.getVOrientation()[k]+=K2[k][l+3]*Theta.getVCenter()[l];
                        df.getVCenter()[k]+=K2[k+3][l]*Theta.getVOrientation()[l]; df.getVCenter()[k]+=K2[k+3][l+3]*Theta.getVCenter()[l];
                    }
                Force[i].getVOrientation()+=df.getVOrientation(); Force[i].getVCenter()+=df.getVCenter();

//qDebug()<<"thetaj"<<Theta.getVOrientation()[0]<<","<<Theta.getVOrientation()[1]<<","<<Theta.getVOrientation()[2]<<","<<Theta.getVCenter()[0]<<","<<Theta.getVCenter()[1]<<","<<Theta.getVCenter()[2];
//qDebug()<<"fji"<<df.getVOrientation()[0]<<","<<df.getVOrientation()[1]<<","<<df.getVOrientation()[2]<<","<<df.getVCenter()[0]<<","<<df.getVCenter()[1]<<","<<df.getVCenter()[2];

// reciprocal force: Df_j= -Df_i (moved to j)
                df.getVOrientation()-=Crossp*df.getVCenter();
                Force[j].getVOrientation()-=df.getVOrientation(); Force[j].getVCenter()-=df.getVCenter();

//qDebug()<<"fjj"<<-df.getVOrientation()[0]<<","<<-df.getVOrientation()[1]<<","<<-df.getVOrientation()[2]<<","<<-df.getVCenter()[0]<<","<<-df.getVCenter()[1]<<","<<-df.getVCenter()[2];
// new stiffness: Kij=d f_ij/Omega_j=K2 R(T) Kjj=d f_jj/Omega_j=d f_jj/d f_ij Kij
                if(K.size())
                {
                    K3.fill(0);
                    for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                            {
                                K3[l][m]+=K2[l][k] * R[k][m];
                                K3[l][m+3]+=K2[l][k+3] * R[k][m];
                                K3[l+3][m]+=K2[l+3][k] * R[k][m];
                                K3[l+3][m+3]+=K2[l+3][k+3] * R[k][m];
                            }
                    K[i][j]+=K3; K[j][j]-=K3;
// (moved to j)
                    for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                            {
                                K[j][j][l][m] +=Crossp[l][k] * K3[k+3][m];
                                K[j][j][l][m+3]+= Crossp[l][k] * K3[k+3][m+3];
                            }
                }

//action of i
                Multi_Rigid( xjbar,T,xi[i]);
// in ref pos: Theta.dt=jbar.jo-1
                PostoSpeed( Theta,xiref[i],xjbar);
// in current pos: Df_j= R(jrel_q^1) Kji Theta.dt = K2 Theta.dt
                K2.fill(0);
                for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                        {
                            K2[l][m]+=Kref[j][i][k][m] * R[k][l];
                            K2[l][m+3]+=Kref[j][i][k][m+3] * R[k][l];
                            K2[l+3][m]+=Kref[j][i][k+3][m] * R[k][l];
                            K2[l+3][m+3]+=Kref[j][i][k+3][m+3] * R[k][l];
                        }

                df.getVOrientation().fill(0); df.getVCenter().fill(0);
                for(k=0; k<3; k++) for(l=0; l<3; l++)
                    {
                        df.getVOrientation()[k]+=K2[k][l]*Theta.getVOrientation()[l]; df.getVOrientation()[k]+=K2[k][l+3]*Theta.getVCenter()[l];
                        df.getVCenter()[k]+=K2[k+3][l]*Theta.getVOrientation()[l]; df.getVCenter()[k]+=K2[k+3][l+3]*Theta.getVCenter()[l];
                    }
                Force[j].getVOrientation()+=df.getVOrientation(); Force[j].getVCenter()+=df.getVCenter();

//qDebug()<<"thetai"<<Theta.getVOrientation()[0]<<","<<Theta.getVOrientation()[1]<<","<<Theta.getVOrientation()[2]<<","<<Theta.getVCenter()[0]<<","<<Theta.getVCenter()[1]<<","<<Theta.getVCenter()[2];
//qDebug()<<"fij"<<df.getVOrientation()[0]<<","<<df.getVOrientation()[1]<<","<<df.getVOrientation()[2]<<","<<df.getVCenter()[0]<<","<<df.getVCenter()[1]<<","<<df.getVCenter()[2];
// reciprocal force: Df_j= -Df_i (moved to j)
                df.getVOrientation()+=Crossp*df.getVCenter();// OiOj=xi[i].t-xi[j].t; cr=cross(df.getVCenter(),OiOj); df.getVOrientation()+=cr;
                Force[i].getVOrientation()-=df.getVOrientation(); Force[i].getVCenter()-=df.getVCenter();

//qDebug()<<"fii"<<-df.getVOrientation()[0]<<","<<-df.getVOrientation()[1]<<","<<-df.getVOrientation()[2]<<","<<-df.getVCenter()[0]<<","<<-df.getVCenter()[1]<<","<<-df.getVCenter()[2];
// new stiffness: Kji=d f_ji/Omega_i=K2 R(T) Kii=d f_i/Omega_i=d f_ii/d f_ji Kji
                if(K.size())
                {
                    K3.fill(0);
                    for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                            {
                                K3[l][m]+=K2[l][k] * R[k][m];
                                K3[l][m+3]+=K2[l][k+3] * R[k][m];
                                K3[l+3][m]+=K2[l+3][k] * R[k][m];
                                K3[l+3][m+3]+=K2[l+3][k+3] * R[k][m];
                            }
                    K[j][i]+=K3; K[i][i]-=K3;
// (moved to j)
                    for(l=0; l<3; l++) for(m=0; m<3; m++) for(k=0; k<3; k++)
                            {
                                K[i][i][l][m] -=Crossp[l][k] * K3[k+3][m];
                                K[i][i][l][m+3] -= Crossp[l][k] * K3[k+3][m+3];
                            }
                }
            }
    }

    /*
    // test momentum conservation
    df.getVOrientation().fill(0); df.getVCenter().fill(0);
    for(i=0;i<nbDOF;i++)
    {
    OiOj=-xi[i].t; cr=cross(Force[i].getVCenter(),OiOj);
    df.getVOrientation()+=Force[i].getVOrientation()+cr;
    df.getVCenter()+=Force[i].getVCenter();
    }
    qDebug()<<"w:"<<df.getVOrientation().norm()<<"v:"<<df.getVCenter().norm();*/
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::XItoQ ( DUALQUAT& q, const Coord& xi )
{
    // xi: quat(a,b,c,w),tx,ty,tz
    // qi: quat(a,b,c,w),1/2quat(t.q)
    const Quat& roti = xi.getOrientation();
    const Vec3& ti = xi.getCenter();

    q.q0[0]=roti[0];
    q.q0[1]=roti[1];
    q.q0[2]=roti[2];
    q.q0[3]=roti[3];
    q.qe[3]= ( - ( roti[0]*ti[0]+roti[1]*ti[1]+roti[2]*ti[2] ) ) /2.;
    q.qe[0]= ( ti[1]*roti[2]-roti[1]*ti[2]+roti[3]*ti[0] ) /2.;
    q.qe[1]= ( ti[2]*roti[0]-roti[2]*ti[0]+roti[3]*ti[1] ) /2.;
    q.qe[2]= ( ti[0]*roti[1]-roti[0]*ti[1]+roti[3]*ti[2] ) /2.;
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::Multi_Rigid( Coord& x1x2, const Coord& x1, const Coord& x2)
{
    Multi_Q( x1x2.getOrientation(),x1.getOrientation(),x2.getOrientation());
    Transform_Q( x1x2.getCenter(),x2.getCenter(),x1.getOrientation());
    x1x2.getCenter()+=x1.getCenter();
}

template<class DataTypes>
void FrameSpringForceField2<DataTypes>::QtoXI ( Coord& xi, const DUALQUAT& q )
{
// xi: quat(a,b,c,w),tx,ty,tz
// qi: quat(a,b,c,w),1/2quat(t.q)
    xi.getOrientation()[0]=q.q0[0];
    xi.getOrientation()[1]=q.q0[1];
    xi.getOrientation()[2]=q.q0[2];
    xi.getOrientation()[3]=q.q0[3];

    xi.getCenter()[0]=2.*(-q.q0[2]*q.qe[1]+q.q0[1]*q.qe[2]+q.q0[3]*q.qe[0]);
    xi.getCenter()[1]=2.*(-q.q0[0]*q.qe[2]+q.q0[2]*q.qe[0]+q.q0[3]*q.qe[1]);
    xi.getCenter()[2]=2.*(-q.q0[1]*q.qe[0]+q.q0[0]*q.qe[1]+q.q0[3]*q.qe[2]);
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
    Omega.getVOrientation()[0]=Om[0]*W;
    Omega.getVOrientation()[1]=Om[1]*W;
    Omega.getVOrientation()[2]=Om[2]*W;
    Omega.getVCenter()=xi2.getCenter()-xi.getCenter();
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

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
