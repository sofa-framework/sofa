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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef FLEXIBLE_MuscleMaterialBlock_H
#define FLEXIBLE_MuscleMaterialBlock_H


#include "../material/BaseMaterial.h"
#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{

template<class T>
class MuscleMaterialBlock : public BaseMaterialBlock<T> {};


//////////////////////////////////////////////////////////////////////////////////
////  E311
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class MuscleMaterialBlock< E311(_Real) >:
    public  BaseMaterialBlock< E311(_Real) >
{
    public:
    typedef E311(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
          * DOFs: strain computed with corotational mapping : E
          *
          * fiber stretch :  lambda = E + 1
          * normalized stretch :  lambdaN  =  lambda/Lambda0
          * 2nd PK stress :  f = -vol*sigmaMax*1/Lambda0 * (fpassive + a.factive) = c * (fpassive + a.factive)
          *
          * passive part:
          *    -  fpassive = 0                                    lambdaN <= 1
          *    -  fpassive = P1.(exp(P2*(lambdaN-1))-1)           1 < lambdaN <= lambdaL
          *              k = df/dE = c*P2/Lambda0*( fpassive + P1 )
          *    -  fpassive = P3*lambdaN + P4                      lambdaL < lambdaN
          *                = P1*P2*exp((lambdaL-1)*P2)*lambdaN + P1.(exp((lambdaL-1)*P2)*(1-lambdaL*P2)-1)    to ensure CO and C1 continuity
          *              k = c*P3/Lambda0
          * active part:
          *    -  factive = 9*(lambdaN-0.4)^2                   lambdaN <= 0.6
          *             k = 18*c*a(lambdaN-0.4)/Lambda0
          *    -  factive = 1-4*(1-lambdaN)^2                   0.6 < lambdaN <= 1.4
          *             k = 8*c*a*(1-lambdaN)/Lambda0
          *    -  factive = 9*(lambdaN-1.6)^2                   1.4 < lambdaN
          *             k = 18*c*a(lambdaN-1.6)/Lambda0
          *
          * for details, see  "A 3D model of muscle reveals the causes of nonuniform strains in the biceps brachii", Blemker et al., 2005
          */


    static const bool constantK=false;

    Real C; ///< -vol*sigmaMax/Lambda0
    Real P1;
    Real P2;
    Real P3;
    Real P4;
    Real Lambda0;
    Real LambdaL;
    Real A;

    mutable Real K;

    void init(const Real p1,const Real p2,const Real l0,const Real ll,const Real a,const Real sigmaMax)
    {
        K=0;
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        P1 = p1;
        P2 = p2;
        LambdaL = ll;
        P3 = P1*P2*exp((LambdaL-1.)*P2);
        P4 = P3*(1./P2-LambdaL)-P1;
        Lambda0 = l0;
        A = a;
        C = - sigmaMax * vol / l0;
    }

    Real getPotentialEnergy(const Coord& /*x*/) const
    {
        // TODO
        return 0;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/) const
    {
        Real lambdaN = (x.getStrain()[0]+1.)/Lambda0;

        Real fpassive = 0 , kpassive = 0;
        if(lambdaN>1.0)
        {
            if(lambdaN<=LambdaL)
            {
                fpassive = P1*(exp(P2*(lambdaN-1.))-1.);
                kpassive = P2/Lambda0*( fpassive + P1 );
            }
            else
            {
                fpassive = P3*lambdaN + P4;
                kpassive = P3/Lambda0;
            }
        }

        Real factive= 0 , kactive = 0;
        if(lambdaN<=0.6)
        {
            factive = 9.*(lambdaN-0.4)*(lambdaN-0.4);
            kactive = 18*(lambdaN-0.4)/Lambda0;
        }
        else  if(lambdaN<=1.4)
        {
            factive = 1-4.*(1.-lambdaN)*(1.-lambdaN) ;
            kactive = 8.*(1.-lambdaN)/Lambda0;
        }
        else
        {
            factive = 9.*(lambdaN-1.6)*(lambdaN-1.6);
            kactive = 18*(lambdaN-1.6)/Lambda0;
        }

//        std::cout<<"lambdaN "<<lambdaN<<" fpassive "<<fpassive<<" kpassive "<<kpassive<<std::endl;

        f.getStrain()[0] += C*(fpassive + A*factive);
        K = C*(kpassive + A*kactive);
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const double& kfactor, const double& /*bfactor*/ ) const
    {
        df.getStrain()+=K*dx.getStrain()*kfactor;
    }

    MatBlock getK() const
    {
        MatBlock mK;
        mK[0][0]=K;
        return mK;
    }

    MatBlock getC() const
    {
        MatBlock C;
        if(K) C[0][0]=-1./K;
        else C[0][0]=-std::numeric_limits<Real>::max();
        return C;
    }

    MatBlock getB() const
    {
        return MatBlock();
    }

};







} // namespace defaulttype
} // namespace sofa



#endif

