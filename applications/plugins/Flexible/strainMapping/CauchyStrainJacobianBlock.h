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
#ifndef FLEXIBLE_CauchyStrainJacobianBlock_H
#define FLEXIBLE_CauchyStrainJacobianBlock_H

#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block for CauchyStrainMapping */
template<class TIn,class TOut>
class CauchyStrainJacobianBlock :
    public  BaseJacobianBlock< TIn , TOut >
{
public:
    typedef TIn In;
    typedef TOut Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { order = Out::order };
    enum { frame_size = spatial_dimensions*material_dimensions };

    /**
    Mapping:
        - \f$ E = [ F + F^T ]/2 - I  \f$
        - \f$ E_k = [ F_k + F_k^T ]/2  \f$
    where:
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [ dF + dF^T  ]/2 \f$
        - \f$  dE_k = [ dF_k + dF_k^T ]/2 \f$
      */

    static const bool constant=true;

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getStrain() += StrainMatToVoigt( data.getF() ); // ( F + Ft ) * 0.5
        for(unsigned int j=0; j<material_dimensions; j++) result.getStrain()[j]-=1.;

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += StrainMatToVoigt( data.getGradientF( k ) ); // (T+Tt)*0.5
            }
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += StrainMatToVoigt( data.getF() );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += StrainMatToVoigt( data.getGradientF(k) );
            }
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        StrainMat s = StressVoigtToMat( data.getStrain() );
        for( unsigned int i=0; i<material_dimensions; i++ )  for(unsigned int j=0; j<material_dimensions; j++)       result.getF()[i][j] += s[i][j];

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                s = StressVoigtToMat( data.getStrainGradient(k) );
                for( unsigned int i=0; i<material_dimensions; i++ )  for(unsigned int j=0; j<material_dimensions; j++)       result.getGradientF(k)[i][j] += s[i][j];
            }
        }
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        typedef Eigen::Matrix<Real,strain_size,frame_size,Eigen::RowMajor> JBlock;
        Frame Id;  for( unsigned int i=0; i<material_dimensions; i++ ) Id[i][i]=(Real)1.;
        JBlock J = this->assembleJ(Id);
        eB.block(0,0,strain_size,frame_size) = J;

        if( order > 0 )
        {
            // order 1
            unsigned int offsetE=strain_size;
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                eB.block(offsetE,(k+1)*frame_size,strain_size,frame_size) = J;
                offsetE+=strain_size;
            }
        }
        return B;
    }

    KBlock getK(const OutDeriv& /*childForce*/, bool=false)    { return KBlock(); }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const SReal& /*kfactor */)    { }

};


} // namespace defaulttype
} // namespace sofa



#endif
