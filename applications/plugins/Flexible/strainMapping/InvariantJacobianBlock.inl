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
#ifndef FLEXIBLE_InvariantJacobianBlock_INL
#define FLEXIBLE_InvariantJacobianBlock_INL

#include "../strainMapping/InvariantJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <sofa/helper/PolarDecompose.h>

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F331(type)  DefGradientTypes<3,3,1,type>
#define F332(type)  DefGradientTypes<3,3,2,type>
#define I331(type)  BaseStrainTypes<3,3,1,type>
#define I332(type)  BaseStrainTypes<3,3,2,type>
#define I333(type)  BaseStrainTypes<3,3,3,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////

/// return dest = det(from).from^-T and det(from)
template<class real>
inline real getDerivDeterminant(Mat<3,3,real>& dest, const Mat<3,3,real>& from)
{
    real det=determinant(from);

    dest(0,0)= (from(1,1)*from(2,2) - from(2,1)*from(1,2));
    dest(0,1)= (from(1,2)*from(2,0) - from(2,2)*from(1,0));
    dest(0,2)= (from(1,0)*from(2,1) - from(2,0)*from(1,1));
    dest(1,0)= (from(2,1)*from(0,2) - from(0,1)*from(2,2));
    dest(1,1)= (from(2,2)*from(0,0) - from(0,2)*from(2,0));
    dest(1,2)= (from(2,0)*from(0,1) - from(0,0)*from(2,1));
    dest(2,0)= (from(0,1)*from(1,2) - from(1,1)*from(0,2));
    dest(2,1)= (from(0,2)*from(1,0) - from(1,2)*from(0,0));
    dest(2,2)= (from(0,0)*from(1,1) - from(1,0)*from(0,1));

    return det;
}


//////////////////////////////////////////////////////////////////////////////////
////  F331 -> I331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class InvariantJacobianBlock< F331(InReal) , I331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , I331(OutReal) >
{
public:
    typedef F331(InReal) In;
    typedef I331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::MaterialFrame MaterialFrame;  ///< Matrix representing a deformation gradient
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;

    /**
    Mapping:
        - \f$ I1 = trace(C) \f$
        - \f$ I2 = ( trace(C^2)+trace(C)^2 )/2 \f$
        - \f$ J = det(F) \f$
    where:
        - \f$  C=F^TF \f$ is the right Cauchy deformation tensor
    Jacobian:
        - \f$  dI1 = trace(dF^T F + F^T dF ) = 2 * sum F_i.dF_i \f$
        - \f$  dI2 = 2 * sum ( F(I1*Id - C) )_i dF_i \f$
        - \f$  dJ = J sum (F^-T)_i dF_i \f$
    */

    static const Real MIN_DETERMINANT=0.2;

    static const bool constantJ=false;
    bool deviatoric;
    Real Jm23; Real Jm43; Real Jm53; Real Jm73; ///< stored variables to handle deviatoric invariants

    InCoord F;   ///< =  store deformation gradient to compute J
    MaterialFrame dI2;   ///<
    MaterialFrame dJ;   ///<

    void addapply( OutCoord& result, const InCoord& data )
    {
        F=data;
        Real detF=getDerivDeterminant(dJ, F.getMaterialFrame());
        if ( detF<=MIN_DETERMINANT)
        {
            //      dJ*=detF/MIN_DETERMINANT; if(detF<0) dJ*=-1;
            detF = MIN_DETERMINANT;   // clamp
        }

        StrainMat C=F.getMaterialFrame().multTranspose( F.getMaterialFrame() );

        Real I1 = C[0][0] + C[1][1] + C[2][2];
        Real I2 = C[0][0]*(C[1][1] + C[2][2]) + C[1][1]*C[2][2] - C[0][1]*C[0][1] - C[0][2]*C[0][2] - C[1][2]*C[1][2];

        for(unsigned int j=0; j<material_dimensions; j++) C[j][j]-=I1;
        dI2=-F.getMaterialFrame()*C*(Real)2.;

        if(deviatoric)
        {
            Jm23=pow(detF,-(Real)2./(Real)3.); Jm43=Jm23*Jm23;
            I1*=Jm23; I2*=Jm43;
            Jm53=(Real)2.*I1/(detF*(Real)3.); Jm73=(Real)4.*I2/(detF*(Real)3.);
        }

        result.getStrain()[0]+= I1;
        result.getStrain()[1]+= I2;
        result.getStrain()[2]+= detF;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        Real di1 =  scalarProduct(F.getMaterialFrame(),data.getMaterialFrame())*(Real)2.;
        Real di2 =  scalarProduct(dI2,data.getMaterialFrame());
        Real dj =   scalarProduct(dJ,data.getMaterialFrame());

        if(deviatoric)
        {
            di1 = di1*Jm23 - dj*Jm53;
            di2 = di2*Jm43 - dj*Jm73;
        }

        result.getStrain()[0] +=  di1;
        result.getStrain()[1] +=  di2;
        result.getStrain()[2] +=  dj;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        if(deviatoric)
        {
            result.getMaterialFrame() += (F.getMaterialFrame()*(Real)2.*Jm23 - dJ*Jm53)*data.getStrain()[0];
            result.getMaterialFrame() += (dI2*Jm43 - dJ*Jm73)*data.getStrain()[1];
        }
        else
        {
            result.getMaterialFrame() += F.getMaterialFrame()*data.getStrain()[0]*(Real)2.;
            result.getMaterialFrame() += dI2*data.getStrain()[1];
        }
        result.getMaterialFrame() += dJ*data.getStrain()[2];
    }

    MatBlock getJ()
    {
        MatBlock B;
        if(deviatoric)
        {
            for(unsigned int j=0; j<frame_size; j++)      B[0][j+spatial_dimensions] +=  *(&F.getMaterialFrame()[0][0]+j)*(Real)2.*Jm23 - *(&dJ[0][0]+j)*Jm53;
            for(unsigned int j=0; j<frame_size; j++)      B[1][j+spatial_dimensions] +=  *(&dI2[0][0]+j)*Jm43 - *(&dJ[0][0]+j)*Jm73;
        }
        else
        {
            for(unsigned int j=0; j<frame_size; j++)      B[0][j+spatial_dimensions] +=  *(&F.getMaterialFrame()[0][0]+j)*(Real)2.;
            for(unsigned int j=0; j<frame_size; j++)      B[1][j+spatial_dimensions] +=  *(&dI2[0][0]+j);
        }
        for(unsigned int j=0; j<frame_size; j++)      B[2][j+spatial_dimensions] +=  *(&dJ[0][0]+j);
        return B;
    }
};



} // namespace defaulttype
} // namespace sofa



#endif
