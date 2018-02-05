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
#ifndef FLEXIBLE_LinearStrainJacobianBlock_H
#define FLEXIBLE_LinearStrainJacobianBlock_H

#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block for LinearStrainMapping */
template<class TStrain>
class LinearStrainJacobianBlock : public  BaseJacobianBlock< TStrain , TStrain >
{
public:

    typedef BaseJacobianBlock<TStrain,TStrain> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename TStrain::StrainMat StrainMat;  ///< Matrix representing a strain
    typedef typename TStrain::StrainVec StrainVec;  ///< Vec representing a strain (Voigt notation)
    enum { strain_size = TStrain::strain_size };
    enum { order = TStrain::order };
    enum { spatial_dimensions = TStrain::spatial_dimensions };

    /**
    Mapping:
        - \f$ E = w.Ein  \f$
        - \f$ E_k = w.Ein_k  \f$
    where:
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = w.dEin \f$
        - \f$  dE_k = w.dEin_k  \f$
      */

    static const bool constant=true;
    Real w;      ///< =   w         =  dE/dEin


    void init(  const Real& _w )
    {
        w=_w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getStrain() += data.getStrain()*w;

        if( order > 0 )
        {
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += data.getStrainGradient(k)*w;
            }
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() +=  data.getStrain()*w;

        if( order > 0 )
        {
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += data.getStrainGradient(k)*w;
            }
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getStrain() +=  data.getStrain()*w;

        if( order > 0 )
        {
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += data.getStrainGradient(k)*w;
            }
        }
    }

    MatBlock getJ()
    {
        return MatBlock::s_identity*w;
    }

    KBlock getK(const OutDeriv& /*childForce*/, bool=false)    { return KBlock(); }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const SReal& /*kfactor */)    { }

};


} // namespace defaulttype
} // namespace sofa



#endif
