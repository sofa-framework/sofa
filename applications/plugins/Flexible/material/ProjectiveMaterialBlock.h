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

#ifndef FLEXIBLE_ProjectiveMaterialBlock_H
#define FLEXIBLE_ProjectiveMaterialBlock_H


#include "../material/BaseMaterial.h"
#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{

template<class T>
class ProjectiveMaterialBlock : public BaseMaterialBlock<T> {};


//////////////////////////////////////////////////////////////////////////////////
////  F311
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class ProjectiveMaterialBlock< F331(_Real) >:
    public  BaseMaterialBlock< F331(_Real) >
{
    public:
    typedef F331(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      Energy:  W = vol.young/2 || F-P(F) ||² where F is the deformation gradient, P(F) a projection
      force(1st piola kirchoff):   f = -young.vol.(F-R)   - viscosity.vol.strainRate
                                   df = (- young.vol.kfactor - viscosity.vol.bfactor) dF
        */

    static const bool constantK=true;

    mutable Real K;
    mutable Real B;

    typedef typename T::Frame Affine;
    mutable Affine P;

    void init(const Real youngModulus, const Real viscosity)
    {
        K=B=0;

        if(this->volume)
        {
            K=(*this->volume)[0]*youngModulus;
            B=(*this->volume)[0]*viscosity;
        }
    }


    // to do: add other types of projection
    // cf. 'projective dynamics' paper
    void project(const Affine& F) const
    {
        helper::Decompose<Real>::polarDecomposition_stable( F, P );
    }



    Real getPotentialEnergy(const Coord& x) const
    {
        this->project(x.getF());
        Real W=0;
        for (int i=0; i<P.nbLines; i++) for (int j=0; j<P.nbCols; j++) W += (x.getF()[i][j]-P[i][j])* (x.getF()[i][j]-P[i][j]);
        return W*K*0.5;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v) const
    {
        this->project(x.getF());
        f.getF()-= K*(x.getF()- P);
        f.getF()-= B*v.getF();
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const SReal& kfactor, const SReal& bfactor ) const
    {
        df.getF() -= dx.getF()*(kfactor*K+bfactor*B);
    }

    MatBlock getK() const
    {
        MatBlock mK; mK.clear();
        for (int i=0; i<mK.nbLines; i++) mK[i][i]=-K;
        return mK;
    }

    MatBlock getC() const
    {
        MatBlock C; C.clear();
        for (int i=0; i<C.nbLines; i++) C[i][i]=1./K;
        return C;
    }

    MatBlock getB() const
    {
        MatBlock mB; mB.clear();
        for (int i=0; i<mB.nbLines; i++) mB[i][i]=-B;
        return mB;
    }

};







} // namespace defaulttype
} // namespace sofa



#endif

