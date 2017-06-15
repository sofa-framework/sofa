/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef FLEXIBLE_GreenStrainJacobianBlock_H
#define FLEXIBLE_GreenStrainJacobianBlock_H

#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block for GreenStrainMapping */
template<class TIn,class TOut>
class GreenStrainJacobianBlock :
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
        - \f$ E = [ F^T.F - I ]/2  \f$
        - \f$ E_k = [ F_k^T.F + F^T.F_k ]/2  \f$
        - \f$ E_jk = E_kj = [ F_k^T.F_j + F_j^T.F_k ]/2  \f$
        - \f$ E_kk = [ F_k^T.F_k ]/2  \f$
    where:
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [ F^T.dF + dF^T.F ]/2 \f$
        - \f$  dE_k = [ F_k^T.dF + dF^T.F_k + dF_k^T.F + F^T.dF_k]/2 \f$
        - \f$  dE_jk = [ F_k^T.dF_j + dF_j^T.F_k + dF_k^T.F_j + F_j^T.dF_k]/2 \f$
        - \f$  dE_kk = [ F_k^T.dF_k + dF_k^T.F_k ]/2 \f$
      */

    static const bool constant=false;

    InCoord F;   ///< =  store deformation gradient to compute J

    void addapply( OutCoord& result, const InCoord& data )
    {
        F=data;
        // order 0
        StrainMat strainmat=F.getF().multTranspose( F.getF() );
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=1.;
        strainmat*=(Real)0.5;
        result.getStrain() += StrainMatToVoigt( strainmat );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                strainmat = F.getF().multTranspose( F.getGradientF(k) );
                result.getStrainGradient(k) += StrainMatToVoigt( strainmat );
            }

            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int j=k+1; j<spatial_dimensions; j++)
                    {
                        strainmat =F.getGradientF(j).multTranspose( F.getGradientF(k) );
                        result.getStrainHessian(j,k) += StrainMatToVoigt( strainmat );
                    }
                for(unsigned int k=0; k<spatial_dimensions; k++)
                {
                    strainmat =F.getGradientF(k).multTranspose( F.getGradientF(k) );
                    strainmat*=(Real)0.5;
                    result.getStrainHessian(k,k) += StrainMatToVoigt( strainmat );
                }
            }
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += StrainMatToVoigt( F.getF().multTranspose( data.getF() ) );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                StrainMat strainmat = F.getF().multTranspose( data.getGradientF(k) ) + F.getGradientF(k).multTranspose( data.getF() );
                result.getStrainGradient(k) += StrainMatToVoigt( strainmat );
            }

            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int j=0; j<spatial_dimensions; j++)
                    {
                        StrainMat strainmat = F.getGradientF(k).multTranspose( data.getGradientF(j) );
                        result.getStrainHessian(j,k) += StrainMatToVoigt( strainmat );
                    }
            }
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += F.getF()*StressVoigtToMat( data.getStrain() );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                StrainMat strainmat=StressVoigtToMat( data.getStrainGradient(k) );
                result.getF() += F.getGradientF(k)*strainmat;
                result.getGradientF(k) += F.getF()*strainmat;
            }

            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int j=k; j<spatial_dimensions; j++)
                    {
                        StrainMat strainmat=StressVoigtToMat( data.getStrainHessian(k,j) );
                        result.getGradientF(k) += F.getGradientF(j)*strainmat;
                        if(j!=k) result.getGradientF(j) += F.getGradientF(k)*strainmat;
                    }
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
        JBlock J = this->assembleJ(F.getF());
        eB.block(0,0,strain_size,frame_size) = J;

        if( order > 0 )
        {
            // order 1
            Vec<spatial_dimensions,JBlock> Jgrad;
            for(unsigned int k=0; k<spatial_dimensions; k++) Jgrad[k]= this->assembleJ(F.getGradientF(k));
            unsigned int offsetE=strain_size;
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                eB.block(offsetE,0,strain_size,frame_size) = Jgrad[k];
                eB.block(offsetE,(k+1)*frame_size,strain_size,frame_size) = J;
                offsetE+=strain_size;
            }

            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int j=k; j<spatial_dimensions; j++)
                    {
                        eB.block(offsetE,(j+1)*frame_size,strain_size,frame_size) = Jgrad[k];
                        if(j!=k) eB.block(offsetE,(k+1)*frame_size,strain_size,frame_size) = Jgrad[j];
                        offsetE+=strain_size;
                    }
            }
        }
        return B;
    }

    KBlock getK(const OutDeriv& childForce, bool /*stabilization*/=false)
    {
        KBlock K = KBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,In::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eK(&K[0][0]);

        // order 0
        StrainMat sigma=StressVoigtToMat( childForce.getStrain() );
        typedef Eigen::Map<Eigen::Matrix<Real,material_dimensions,material_dimensions,Eigen::RowMajor> > KBlock;
        KBlock s(&sigma[0][0]);
        for(unsigned int j=0; j<spatial_dimensions; j++)
            eK.block(j*material_dimensions,j*material_dimensions,material_dimensions,material_dimensions) += s;

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                sigma=StressVoigtToMat( childForce.getStrainGradient(k) );
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<spatial_dimensions; j++)
                    {
                        eK.block(j*material_dimensions+i*material_dimensions*spatial_dimensions,j*material_dimensions,material_dimensions,material_dimensions) += s;
                        eK.block(j*material_dimensions,j*material_dimensions+i*material_dimensions*spatial_dimensions,material_dimensions,material_dimensions) += s;
                    }
            }
            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int l=k; l<spatial_dimensions; l++)
                    {
                        sigma=StressVoigtToMat( childForce.getStrainHessian(k,l) );
                        for(unsigned int j=0; j<spatial_dimensions; j++)
                        {
                            eK.block(j*material_dimensions+k*material_dimensions*spatial_dimensions,j*material_dimensions+l*material_dimensions*spatial_dimensions,material_dimensions,material_dimensions) += s;
                            if(k!=l) eK.block(j*material_dimensions+l*material_dimensions*spatial_dimensions,j*material_dimensions+k*material_dimensions*spatial_dimensions,material_dimensions,material_dimensions) += s;
                        }
                    }
            }
        }
        return K;
    }
    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
    {
        // order 0
        df.getF() += dx.getF()*StressVoigtToMat( childForce.getStrain() )*kfactor;

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                StrainMat strainmat=StressVoigtToMat( childForce.getStrainGradient(k) )*kfactor;
                df.getF() += dx.getGradientF(k)*strainmat;
                df.getGradientF(k) += dx.getF()*strainmat;
            }

            if( order > 1 )
            {
                // order 2
                for(unsigned int k=0; k<spatial_dimensions; k++)
                    for(unsigned int j=k; j<spatial_dimensions; j++)
                    {
                        StrainMat strainmat=StressVoigtToMat( childForce.getStrainHessian(k,j) )*kfactor;
                        df.getGradientF(k) += dx.getGradientF(j)*strainmat;
                        if(j!=k) df.getGradientF(j) += dx.getGradientF(k)*strainmat;
                    }
            }
        }
    }

};


} // namespace defaulttype
} // namespace sofa



#endif
