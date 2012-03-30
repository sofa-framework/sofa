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
#ifndef FLEXIBLE_GreenStrainJacobianBlock_INL
#define FLEXIBLE_GreenStrainJacobianBlock_INL

#include "GreenStrainJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "DeformationGradientTypes.h"
#include "StrainTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F331(type)  DefGradientTypes<3,3,1,type>
#define F332(type)  DefGradientTypes<3,3,2,type>
#define E331(type)  StrainTypes<3,3,1,type>
#define E332(type)  StrainTypes<3,3,2,type>
#define E333(type)  StrainTypes<3,3,3,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////

template<typename Real>
static Mat<6,9,Real> assembleJ(const  Mat<3,3,Real>& f) // 3D
{
    static const unsigned int material_dimensions = 3;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    Mat<strain_size,material_dimensions*material_dimensions,Real> J;
    for(unsigned int j=0; j<material_dimensions; j++)
        for( unsigned int k=0; k<material_dimensions; k++ )
            J[j][j+material_dimensions*k]=f[k][j];
    for( unsigned int k=0; k<material_dimensions; k++ )
    {
        J[3][material_dimensions*k+1]=J[5][material_dimensions*k+2]=f[k][0]*0.5;
        J[3][material_dimensions*k]=J[4][material_dimensions*k+2]=f[k][1]*0.5;
        J[5][material_dimensions*k]=J[4][material_dimensions*k+1]=f[k][2]*0.5;
    }
    return J;
}

template<typename Real>
static Mat<3,4,Real> assembleJ(const  Mat<2,2,Real>& f) // 2D
{
    static const unsigned int material_dimensions = 2;
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
    Mat<strain_size,material_dimensions*material_dimensions,Real> J;
    for(unsigned int j=0; j<material_dimensions; j++)
        for( unsigned int k=0; k<material_dimensions; k++ )
            J[j][j+material_dimensions*k]=f[k][j];
    for( unsigned int k=0; k<material_dimensions; k++ )
    {
        J[material_dimensions][material_dimensions*k+1]=f[k][0]*0.5;
        J[material_dimensions][material_dimensions*k]=f[k][1]*0.5;
    }
    return J;
}

//////////////////////////////////////////////////////////////////////////////////
////  F331 -> E331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class GreenStrainJacobianBlock< F331(InReal) , E331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , E331(OutReal) >
{
public:
    typedef F331(InReal) In;
    typedef E331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = In::material_dimensions };
    typedef typename In::MaterialFrame MaterialFrame;  ///< Matrix representing a deformation gradient
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };

    /**
    Mapping:   \f$ E = (E = [F^T.F - I ]/2  \f$
    Jacobian:    \f$  dE = [ F^T.dF + dF^T.F ]/2 \f$
      */

    InCoord F;   ///< =  store deformation gradient to compute J

    void addapply( OutCoord& result, const InCoord& data )
    {
        F=data;
        MaterialFrame strainmat=F.getMaterialFrame().multTranspose( F.getMaterialFrame() );
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=1.;
        strainmat*=(Real)0.5;
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        MaterialFrame strainmat=F.getMaterialFrame().multTranspose( data.getMaterialFrame() );
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getMaterialFrame() += F.getMaterialFrame()*VoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        Mat<strain_size,material_dimensions*material_dimensions,Real> J = assembleJ(F.getMaterialFrame());
        MatBlock B;
        for(unsigned int j=0; j<strain_size; j++)
            memcpy(&B[j][spatial_dimensions],&J[j][0],material_dimensions*material_dimensions*sizeof(Real));
        return B;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif
