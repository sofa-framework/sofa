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
          * vol = l0
          * fiber stretch :  lambda = l/l0 = E + 1
          * normalized stretch :  lambdaN  =  l/lopt = lambda/Lambda0
          * normalized strain :  EN = lambdaN - 1
          *
          *   f = -vol*sigmaMax*a*fl*fv
          *   with fl = exp{-EN^2/b^2}
          *        fv = Vsh(Vmax + lopt.ENdot)/(Vsh.Vmax-lopt.ENdot)    with   Vmax = vm(1-ver(1-a.fl))  and  lopt.ENdot = ldot = Edot.l0
          *   k = -2*f*EN/(b^2*Lambda0)
          *   b = -vol*sigmaMax*a*fl * Vmax.(Vsh+1)*Vsh/(Vsh.Vmax-lopt.ENdot)^2
          */


    static const bool constantK=false;

    Real A;
    Real C; ///< -vol*sigmaMax*a
    Real D; ///< -2/(b*Lambda0)
    Real Lambda0;
    Real B2;
    Real Vvm;
    Real Ver;
    Real Vsh;
    Real L0;

    mutable Real K;
    mutable Real B;

    void init(const Real lambda0,const Real sigmaMax,const Real a,const Real b,const Real vvm,const Real ver,const Real vsh)
    {
        K=0;
        L0=1.;
        if(this->volume) L0=(*this->volume)[0];

        Lambda0 = lambda0;
        A = a;
        B2 = b*b;
        C = - sigmaMax * L0 * a;
        D = -2./(B2*lambda0);
        Vvm = vvm;
        Ver = ver;
        Vsh = vsh;
    }

    Real getPotentialEnergy(const Coord& /*x*/) const
    {
        // TODO
        return 0;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v) const
    {
        Real EN = (x.getStrain()[0]+1.)/Lambda0 - 1.;
        Real Fl = exp(-EN*EN/B2);
        Real Vmax = Vvm*(1. - Ver*(1. - A*Fl ) );
        Real ldot = v.getStrain()[0]*L0;
        Real fact = (Vsh*Vmax-ldot);
        if(fact==0) fact=1E-10;
        Real Fv = Vsh*(Vmax + ldot)/fact;

        Real F = C*Fl*Fv;
//        std::cout<<"ldot,vmax = "<<ldot<<","<<Vmax<<std::endl;
//        std::cout<<"Fv = "<<Fv<<std::endl;

        K = F*D*EN;
        B = C*Fl*Vmax*(Vsh+1.)*Vsh/fact/fact;

        f.getStrain()[0] += F;
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const SReal& kfactor, const SReal& bfactor ) const
    {
        df.getStrain()+=K*dx.getStrain()*kfactor + B*dx.getStrain()*bfactor;
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
        MatBlock mB;
        mB[0][0]=B;
        return mB;

    }

};







} // namespace defaulttype
} // namespace sofa



#endif

