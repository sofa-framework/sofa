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
#define FRAME_FRAMESPRINGFORCEFIELD2_CPP

#include "FrameSpringForceField2.inl"
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include "AffineTypes.h"
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(FrameSpringForceField2)

using namespace sofa::defaulttype;


// Register in the Factory
int FrameSpringForceField2Class = core::RegisterObject("Springs between frames.")
#ifndef SOFA_FLOAT
        .add< FrameSpringForceField2<Rigid3dTypes> >()
        .add< FrameSpringForceField2<Affine3dTypes> >()
//.add< FrameSpringForceField2<Quadratic3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
//.add< FrameSpringForceField2<Affine3fTypes> >()
//.add< FrameSpringForceField2<Rigid3fTypes> >()
//.add< FrameSpringForceField2<Quadratic3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameSpringForceField2<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameSpringForceField2<Rigid3fTypes>;
#endif





///////////////////////////////////////////////////////////////////////////////
//                FrameSpringForceField2<Affine3dTypes>                       //
///////////////////////////////////////////////////////////////////////////////

// TODO
template<>
void FrameSpringForceField2<Affine3dTypes>::addDForce(VecDeriv& /*df*/, const VecDeriv& /*dx*/ )
{
    /*
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
      */
}


// TODO
template<>
void FrameSpringForceField2<Affine3dTypes>::updateForce( VecDeriv& /*Force*/, VVMatInxIn& /*K*/, const VecCoord& /*xi*/, const VVMatInxIn& /*Kref*/ )
{
    /*
    // generalized spring joint network based on precomputed FrameHooke stiffness matrices
    VecCoord& xiref = *this->getMState()->getX0();
    int i,j,k,l,m,nbDOF=xi.size();
    double n;
    Coord XI,XJ,XP;
    Deriv Theta,df;
    Vec3 tpref;
    Quat q;
    Mat33 crossi,crossj;
    Mat66 K2,K3;
    Mat33 Rp;

    for(i=0;i<nbDOF;i++)
      {
      for(j=0;j<nbDOF;j++)
          if(i>j)
              {
              tpref=(xiref[j].getCenter() + xiref[i].getCenter())/2.;   // pivot = center

              q=xiref[i].getOrientation();   q[3]*=-1;    Multi_Q(XI.getOrientation(),xi[i].getOrientation(),q);      Transform_Q(XI.getCenter(),tpref-xiref[i].getCenter(),XI.getOrientation());   XI.getCenter()+=xi[i].getCenter();
              q=xiref[j].getOrientation();   q[3]*=-1;    Multi_Q(XJ.getOrientation(),xi[j].getOrientation(),q);      Transform_Q(XJ.getCenter(),tpref-xiref[j].getCenter(),XJ.getOrientation());   XJ.getCenter()+=xi[j].getCenter();

              // in ref pos: Theta.dt=XJ.XI-1
            PostoSpeed(Theta,XI,XJ);

        // mid frame
        XP.getCenter()=(XI.getCenter()+XJ.getCenter())/2.;
        n=Theta.getVOrientation()[0]*Theta.getVOrientation()[0]+Theta.getVOrientation()[1]*Theta.getVOrientation()[1]+Theta.getVOrientation()[2]*Theta.getVOrientation()[2];
        if(n>0) {n=sqrt(n); q[3]=cos(n/4.);  n=sin(n/4.)/n;   q[0]=Theta.getVOrientation()[0]*n; q[1]=Theta.getVOrientation()[1]*n; q[2]=Theta.getVOrientation()[2]*n;}
        else {q[3]=1; q[0]=q[1]=q[2]=0;}

        Multi_Q(XP.getOrientation(),q,XI.getOrientation());
        QtoR(Rp,XP.getOrientation());

        // rotate K in deformed space
        K2=(Kref[i][j]+Kref[j][i])/2.;

        K3.fill(0);
        for(k=0;k<3;k++)  for(l=0;l<3;l++) for(m=0;m<3;m++)
                     {
                     K3[k][l]+=Rp[k][m]*K2[m][l];
                     K3[k][l+3]+=Rp[k][m]*K2[m][l+3];
                     K3[k+3][l]+=Rp[k][m]*K2[m+3][l];
                     K3[k+3][l+3]+=Rp[k][m]*K2[m+3][l+3];
                     }
          K2.fill(0);
                for(k=0;k<3;k++)  for(l=0;l<3;l++) for(m=0;m<3;m++)
                     {
                     K2[k][l]+=K3[k][m]*Rp[l][m];
                     K2[k][l+3]+=K3[k][m+3]*Rp[l][m];
                     K2[k+3][l]+=K3[k+3][m]*Rp[l][m];
                     K2[k+3][l+3]+=K3[k+3][m+3]*Rp[l][m];
             }

        // spring force at p
              df= Deriv();
              for(k=0;k<3;k++) for(l=0;l<3;l++)
                   {
                   df.getVOrientation()[k]+=K2[k][l]*Theta.getVOrientation()[l];     df.getVOrientation()[k]+=K2[k][l+3]*Theta.getVCenter()[l];
                   df.getVCenter()[k]+=K2[k+3][l]*Theta.getVOrientation()[l];   df.getVCenter()[k]+=K2[k+3][l+3]*Theta.getVCenter()[l];
                   }

              // force on i
             Force[i].getVOrientation()+=df.getVOrientation()+cross(XP.getCenter()-xi[i].getCenter(),df.getVCenter()); Force[i].getVCenter()+=df.getVCenter();

        // reciprocal force on j
         Force[j].getVOrientation()-=df.getVOrientation()+cross(XP.getCenter()-xi[j].getCenter(),df.getVCenter()); Force[j].getVCenter()-=df.getVCenter();

              // update K
             if(K.size())
                    {
                    GetCrossproductMatrix(crossi,XP.getCenter()-xi[i].getCenter());
                    GetCrossproductMatrix(crossj,XP.getCenter()-xi[j].getCenter());

                    K3=K2;
                    for(l=0;l<3;l++)  for(m=0;m<3;m++) for(k=0;k<3;k++)
                            {
                            K3[l][m]   += crossi[l][k]*K2[k+3][m];
                            K3[l][m+3] += crossi[l][k]*K2[k+3][m+3];
                            }
            K[i][i]-=K3;  K[i][j]=K3;
                    for(l=0;l<3;l++)  for(m=0;m<3;m++) for(k=0;k<3;k++)
                            {
                            K[i][i][l][m]  += K3[l][k+3]  * crossi[k][m];
                            K[i][i][l+3][m]+= K3[l+3][k+3]* crossi[k][m];
                            K[i][j][l][m]  -= K3[l][k+3]  * crossj[k][m];
                            K[i][j][l+3][m]-= K3[l+3][k+3]* crossj[k][m];
                            }
            K[j][i].transpose(K[i][j]);

            K3=K2;
                    for(l=0;l<3;l++)  for(m=0;m<3;m++) for(k=0;k<3;k++)
                            {
                            K3[l][m]   += crossj[l][k]*K2[k+3][m];
                            K3[l][m+3] += crossj[l][k]*K2[k+3][m+3];
                            }
            K[j][j]-=K3;
                    for(l=0;l<3;l++)  for(m=0;m<3;m++) for(k=0;k<3;k++)
                            {
                            K[j][j][l][m]  += K3[l][k+3]  * crossj[k][m];
                            K[j][j][l+3][m]+= K3[l+3][k+3]* crossj[k][m];
                            }
                      }

              }

         }
    */
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


// TODO
template<>
void FrameSpringForceField2<Affine3dTypes>::PostoSpeed( Deriv& /*Omega*/, const Coord& /*xi*/, const Coord& /*xi2*/)
{
    /*
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
    */
}





///////////////////////////////////////////////////////////////////////////////
//                FrameSpringForceField2<Affine3fTypes>                       //
///////////////////////////////////////////////////////////////////////////////


#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameSpringForceField2<Affine3dTypes>;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameSpringForceField2<Affine3fTypes>;
#endif


/*
#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameSpringForceField2<Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameSpringForceField2<Quadratic3fTypes>;
#endif
*/

} // namespace forcefield

} // namespace component

} // namespace sofa

