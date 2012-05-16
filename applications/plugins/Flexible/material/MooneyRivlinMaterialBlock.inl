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
#ifndef FLEXIBLE_MooneyRivlinMaterialBlock_INL
#define FLEXIBLE_MooneyRivlinMaterialBlock_INL

#include "../material/MooneyRivlinMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"

#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define I331(type)  BaseStrainTypes<3,3,0,type>
#define I332(type)  BaseStrainTypes<3,3,1,type>
#define I333(type)  BaseStrainTypes<3,3,2,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////
////  F331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class MooneyRivlinMaterialBlock< I331(_Real) > :
    public  BaseMaterialBlock< I331(_Real) >
{
public:
    typedef I331(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      * DOFs: sqrt(I1), sqrt(I2), J
      *
      * classic Mooney rivlin
      *     - W = vol* [ C1 ( I1 - 3)  + C2 ( I2 - 3) ]
      *     - f = - 2 [ vol*C1*sqrt(I1) , vol*C2*sqrt(I12) , 0 ]
      *     - df =  - 2 [ vol*C1 , vol*C2 , 0 ]
      */

    Real C1Vol2;  ///<  first coef * volume * 2
    Real C2Vol2;  ///<  second coef * volume * 2

    void init(const Real &C1,const Real &C2)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        C1Vol2=C1*vol*(Real)2.;
        C2Vol2=C2*vol*(Real)2.;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        return C1Vol2*(Real)0.5*(x.getStrain()[0]*x.getStrain()[0]-(Real)3.) + C2Vol2*(Real)0.5*(x.getStrain()[1]*x.getStrain()[1]-(Real)3.);
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/)
    {
        f.getStrain()[0]-=C1Vol2*x.getStrain()[0];
        f.getStrain()[1]-=C2Vol2*x.getStrain()[1];
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const double& kfactor, const double& /*bfactor*/ )
    {
        df.getStrain()[0]-=C1Vol2*dx.getStrain()[0]*kfactor;
        df.getStrain()[1]-=C2Vol2*dx.getStrain()[1]*kfactor;
    }

    MatBlock getK()
    {
        MatBlock K;
        K[0][0]=-C1Vol2;
        K[1][2]=-C2Vol2;
        return K;
    }

    MatBlock getC()
    {
        MatBlock C;
        return C;
    }

    MatBlock getB()
    {
        MatBlock B;
        return B;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif
