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
#ifndef FLEXIBLE_PrincipalStretchesJacobianBlock_H
#define FLEXIBLE_PrincipalStretchesJacobianBlock_H

#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <sofa/helper/decompose.h>

#include <sofa/helper/MatEigen.h>


namespace sofa
{

namespace defaulttype
{



/** Template class used to implement one jacobian block for PrincipalStretchesMapping */
template<class TIn, class TOut>
class PrincipalStretchesJacobianBlock : public BaseJacobianBlock< TIn,TOut >
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
    typedef typename Out::StrainVec StrainVec;  ///< Vec representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<material_dimensions,material_dimensions,Real> MaterialMaterialMat;
    typedef Mat<spatial_dimensions,material_dimensions,Real> SpatialMaterialMat;

    /**
    Mapping:   \f$ E = Ut.F.V\f$
    where:  U/V are the spatial and material rotation parts of F and E is diagonal
    Jacobian:    \f$  dE = Ut.dF.V \f$ Note that dE is still diagonal (no anisotropy possible)
    */

    static const bool constantJ = false;

    SpatialMaterialMat _U;  ///< Spatial Rotation
    MaterialMaterialMat _V; ///< Material Rotation
    StrainVec _S; ///< Principal stretches

    MatBlock _J;

    bool _degenerated;

    void addapply( OutCoord& result, const InCoord& data )
    {
        _degenerated = helper::Decompose<Real>::SVD_stable( data.getF(), _U, _S, _V );

        for( int i=0 ; i<material_dimensions ; ++i )
            result.getStrain()[i] += _S[i] - (Real)1;

        computeJ();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    result.getStrain()[k] += _J[k][i*material_dimensions+j] * data.getF()[i][j];
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    result.getF()[i][j] += _J[k][i*material_dimensions+j] * data.getStrain()[k];
    }

    void computeJ()
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    _J[k][i*material_dimensions+j] = _U[i][k]*_V[j][k];
    }

    MatBlock getJ()
    {
        return _J;
    }

    // TODO
    KBlock getK( const OutDeriv& /*childForce*/ )
    {
        return KBlock();
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _degenerated ) return;

        SpatialMaterialMat dU;
        MaterialMaterialMat dV;
        helper::Decompose<Real>::SVDGradient_dUdV( _U, _S, _V, dx.getF(), dU, dV );
        df.getF() += dU.multDiagonal( childForce.getStrain() ) * _V * kfactor;
        df.getF() += _U.multDiagonal( childForce.getStrain() ) * dV * kfactor;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PrincipalStretchesJacobianBlock_H
