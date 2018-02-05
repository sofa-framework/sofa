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

#ifndef FLEXIBLE_TendonMaterialBlock_H
#define FLEXIBLE_TendonMaterialBlock_H


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
class TendonMaterialBlock : public BaseMaterialBlock<T> {};


//////////////////////////////////////////////////////////////////////////////////
////  E311
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class TendonMaterialBlock< E311(_Real) >:
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
          * 2nd PK stress :  f = -vol * F
          * with :
          *    -  F = 0                                    lambda <= 1
          *    -  F = L1.(exp(L2*E)-1)           1 < lambda <= lambdaL
          *       k = df/dE = -vol*L2*( F + L1 )
          *    -  F = L3*lambda + L4                      1.4 < lambdaN
          *         = L1*L2*exp((lambdaL-1)*L2)*lambda + L1.(exp((lambdaL-1)*L2)*(1-lambdaL*L2)-1)    to ensure CO and C1 continuity
          *       k = -vol*L3
          *
          * for details, see  "A 3D model of muscle reveals the causes of nonuniform strains in the biceps brachii", Blemker et al., 2005
          */


    static const bool constantK=false;

    Real C; ///< -vol
    Real L1;
    Real L2;
    Real L3;
    Real L4;
    Real LambdaL;

    mutable Real K;

    void init(const Real l1,const Real l2,const Real ll)
    {
        K=0;
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        L1 = l1;
        L2 = l2;
        LambdaL = ll;
        L3 = L1*L2*exp((LambdaL-1)*L2);
        L4 = L3*(1./L2-LambdaL)-L1;
        C = - vol;
    }

    Real getPotentialEnergy(const Coord& /*x*/) const
    {
        // TODO
        return 0;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/) const
    {
        Real lambda = x.getStrain()[0]+1.;

        if(lambda>1.0)
        {
            if(lambda<=LambdaL)
            {
                Real F = L1*(exp(L2*(lambda-1.))-1.);
                f.getStrain()[0] += C*F;
                K = C*L2*( F + L1 );
            }
            else
            {
                f.getStrain()[0] += C*(L3*lambda + L4);
                K = C*L3;
            }
        }
        else K=0;
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const SReal& kfactor, const SReal& /*bfactor*/ ) const
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

