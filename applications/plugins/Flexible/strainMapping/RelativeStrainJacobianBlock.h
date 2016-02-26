/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FLEXIBLE_RelativeStrainJacobianBlock_H
#define FLEXIBLE_RelativeStrainJacobianBlock_H

#include "../BaseJacobian.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../helper.h"

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block for RelativeStrainMapping */
template<class TStrain>
class RelativeStrainJacobianBlock : public BaseJacobianBlock<TStrain,TStrain>
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

    static const bool constant = true;
    InCoord offset;
    Real multfactor;

    /**
    Mapping:   ADDITION -> \f$ E_elastic = E_total - E_offset \f$;
               MULTIPLICATION -> \f$ S_elastic = S_total * S_offset^-1 , E_elastic = S_elastic - I \f$
    Jacobian:    \f$  dE = Id \f$
    */


//    void addapply_multiplication( OutCoord& result, const InCoord& data, const InCoord& offset)
//    {
//        StrainMat plasticStrainMat = StrainVoigtToMat( offset.getStrain() ) + StrainMat::s_identity;
//        StrainMat plasticStrainMatInverse; plasticStrainMatInverse.invert( plasticStrainMat );

//        StrainMat elasticStrainMat = ( StrainVoigtToMat( data.getStrain() ) + StrainMat::s_identity ) * plasticStrainMatInverse;
//        StrainVec elasticStrainVec = StrainMatToVoigt( elasticStrainMat - StrainMat::s_identity );

//        result.getStrain() += elasticStrainVec;
//    }

//    void addapply_addition( OutCoord& result, const InCoord& data, const InCoord& offset)
//    {
//        result.getStrain() += (data.getStrain() - offset.getStrain());
//    }
    void init( const InCoord& off, const bool& inverted)
    {
        offset=off;
        multfactor=inverted?(Real)-1.:(Real)1.;
    }

    void addapply( OutCoord& result, const InCoord& data)
    {
        result.getStrain() += (data.getStrain() - offset.getStrain())*multfactor;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data*multfactor;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data*multfactor;
    }

    MatBlock getJ()
    {
        return MatBlock::s_identity*multfactor;
    }

    KBlock getK(const OutDeriv& /*childForce*/, bool=false)
    {
        return KBlock();
    }

    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const SReal& /*kfactor */)
    {
    }

}; // class PlasticStrainJacobianBlock

} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PlasticStrainJacobianBlock_H
