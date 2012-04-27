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
#ifndef FLEXIBLE_VolumePreservationMaterialBlock_INL
#define FLEXIBLE_VolumePreservationMaterialBlock_INL

#include "../material/VolumePreservationMaterialBlock.h"
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
////  F331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class VolumePreservationMaterialBlock< I331(_Real) > :
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
      * method 0:
      *     - W = vol* k/2 ln( J )^2
      *     - f =   [ 0 , 0 , - k*vol*ln(J)/J ]
      *     - df =  [ 0 , 0 , k*vol*(ln(J)-1)/J^2* dJ ]
      * method 1:
      *     - W = vol* k/2 (J-1)^2
      *     - f =   [ 0 , 0 , - k*vol*(J-1) ]
      *     - df =  [ 0 , 0 , - k*vol*dJ ]
      */

    Real KVol;  ///< bulk  * volume
    Real J; ///< store J for stiffness
    unsigned int Method;

    void init(const Real &k,const unsigned int &method)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        KVol=k*vol;
        Method=method;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        if(Method==0) return KVol*log(x.getStrain()[2])*log(x.getStrain()[2])*(Real)0.5;
        else return KVol*(x.getStrain()[2]-(Real)1.)*(x.getStrain()[2]-(Real)1.)*(Real)0.5;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/)
    {
        J=x.getStrain()[2];
        if(Method==0) f.getStrain()[2]-=KVol*log(J)/J;
        else f.getStrain()[2]-=KVol*(J-1);
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& /*bfactor*/ )
    {
        if(Method==0) df.getStrain()[2]+=KVol*(log(J)-(Real)1.)/(J*J)*dx.getStrain()[2]*kfactor;
        else df.getStrain()[2]-=KVol*dx.getStrain()[2]*kfactor;
    }


    MatBlock getK()
    {
        MatBlock K;
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
