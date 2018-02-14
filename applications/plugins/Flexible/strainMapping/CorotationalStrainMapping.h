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
#ifndef SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H

#include <Flexible/config.h>
#include "../strainMapping/BaseStrainMapping.h"
#include "../strainMapping/CorotationalStrainJacobianBlock.inl"

#include <sofa/helper/OptionsGroup.h>

namespace sofa
{
namespace component
{
namespace mapping
{


/** Deformation Gradient to Corotational Lagrangian Strain mapping
 *
 * @author Matthieu Nesme
 *
*/

template <class TIn, class TOut>
class CorotationalStrainMapping : public BaseStrainMappingT<defaulttype::CorotationalStrainJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::CorotationalStrainJacobianBlock<TIn,TOut> BlockType;
    typedef BaseStrainMappingT<BlockType > Inherit;

    SOFA_CLASS(SOFA_TEMPLATE2(CorotationalStrainMapping,TIn,TOut), SOFA_TEMPLATE(BaseStrainMappingT,BlockType ));

    /** @name  Corotational methods
       SMALL = Cauchy strain
       QR = Nesme et al, 2005, "Efficient, Physically Plausible Finite Elements"
       POLAR = Etzmuß et al, 2003, "A Fast Finite Element Solution for Cloth Modelling" ; Muller et al, 2004 "Interactive Virtual Materials"
       SVD = Irving et al, 2004, "Invertible finite elements for robust simulation of large deformation"
       FROBENIUS = Muller et al, 2016, "A Robust Method to Extract the Rotational Part of Deformations"
    */
    //@{
    enum DecompositionMethod { POLAR=0, QR, SMALL, SVD, FROBENIUS, NB_DecompositionMethod };
    Data<helper::OptionsGroup> f_method; ///< Decomposition method
    //@}


    Data<bool> f_geometricStiffness; ///< should geometricStiffness be considered?

    //Pierre-Luc : I added this function to use some functionalities of the mapping component whitout using it as a sofa graph component (protected)
    virtual void initJacobianBlock( helper::vector<BlockType>& jacobianBlock )
    {
        if(this->f_printLog.getValue()==true)
            std::cout << SOFA_CLASS_METHOD << std::endl;

        switch( f_method.getValue().getSelectedId() )
        {
        case SMALL:
        {
            for( size_t i=0 ; i<jacobianBlock.size() ; i++ )
            {
                jacobianBlock[i].init_small();
            }
            break;
        }
        case QR:
        {
            for( size_t i=0 ; i<jacobianBlock.size() ; i++ )
            {
                jacobianBlock[i].init_qr( f_geometricStiffness.getValue() );
            }
            break;
        }
        case POLAR:
        {
            for( size_t i=0 ; i<jacobianBlock.size() ; i++ )
            {
                jacobianBlock[i].init_polar( f_geometricStiffness.getValue() );
            }
            break;
        }
        case SVD:
        {
            for( size_t i=0 ; i<jacobianBlock.size() ; i++ )
            {
                jacobianBlock[i].init_svd( f_geometricStiffness.getValue() );
            }
            break;
        }
        case FROBENIUS:
        {
            for( size_t i=0 ; i<jacobianBlock.size() ; i++ )
            {
                jacobianBlock[i].init_frobenius( f_geometricStiffness.getValue() );
            }
            break;
        }
        }
    }

    virtual void reinit()
    {
        Inherit::reinit();

        switch( f_method.getValue().getSelectedId() )
        {
        case SMALL:
        {
            for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            {
                this->jacobian[i].init_small();
            }
            break;
        }
        case QR:
        {
            for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            {
                this->jacobian[i].init_qr( f_geometricStiffness.getValue() );
            }
            break;
        }
        case POLAR:
        {
            for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            {
                this->jacobian[i].init_polar( f_geometricStiffness.getValue() );
            }
            break;
        }
        case SVD:
        {
            for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            {
                this->jacobian[i].init_svd( f_geometricStiffness.getValue() );
            }
            break;
        }
        case FROBENIUS:
        {
            for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            {
                this->jacobian[i].init_frobenius( f_geometricStiffness.getValue() );
            }
            break;
        }
        }
    }


protected:
    CorotationalStrainMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
        , f_method( initData( &f_method, "method", "Decomposition method" ) )
        , f_geometricStiffness( initData( &f_geometricStiffness, false, "geometricStiffness", "Should geometricStiffness be considered?" ) )
    {
        helper::OptionsGroup Options;
        Options.setNbItems( NB_DecompositionMethod );
        Options.setItemName( SMALL,     "small"     );
        Options.setItemName( QR,        "qr"        );
        Options.setItemName( POLAR,     "polar"     );
        Options.setItemName( SVD,       "svd"       );
        Options.setItemName( FROBENIUS, "frobenius" );
        Options.setSelectedItem( SVD );
        f_method.setValue( Options );
    }

    virtual ~CorotationalStrainMapping() { }

    virtual void applyBlock(Data<typename Inherit::OutVecCoord>& dOut, const Data<typename Inherit::InVecCoord>& dIn, helper::vector<BlockType>& jacobianBlock)
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<":apply"<<std::endl;

        typename Inherit::OutVecCoord& out = *dOut.beginWriteOnly();
        const typename Inherit::InVecCoord&  in  =  dIn.getValue();

        switch( f_method.getValue().getSelectedId() )
        {
        case SMALL:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(jacobianBlock.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                jacobianBlock[i].addapply_small( out[i], in[i] );
            }
            break;
        }
        case QR:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(jacobianBlock.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                jacobianBlock[i].addapply_qr( out[i], in[i] );
            }
            break;
        }
        case POLAR:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(jacobianBlock.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                //std::cout << "applyBlock_polar : Index : " << i << std::endl;
                jacobianBlock[i].addapply_polar( out[i], in[i] );
            }
            break;
        }
        case SVD:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(jacobianBlock.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                jacobianBlock[i].addapply_svd( out[i], in[i] );
            }
            break;
        }
        case FROBENIUS:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(jacobianBlock.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                jacobianBlock[i].addapply_frobenius( out[i], in[i] );
            }
            break;
        }
        }

        dOut.endEdit();
    }

    virtual void apply( const core::MechanicalParams * /*mparams*/ , Data<typename Inherit::OutVecCoord>& dOut, const Data<typename Inherit::InVecCoord>& dIn )
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<":apply"<<std::endl;

        helper::ReadAccessor<Data<typename Inherit::InVecCoord> > inpos (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<typename Inherit::OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
        if(inpos.size()!=outpos.size()) this->resizeOut();

        typename Inherit::OutVecCoord& out = *dOut.beginWriteOnly();
        const typename Inherit::InVecCoord&  in  =  dIn.getValue();

        switch( f_method.getValue().getSelectedId() )
        {
        case SMALL:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_small( out[i], in[i] );
            }
            break;
        }
        case QR:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_qr( out[i], in[i] );
            }
            break;
        }
        case POLAR:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_polar( out[i], in[i] );
            }
            break;
        }
        case SVD:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_svd( out[i], in[i] );
            }
            break;
        }
        case FROBENIUS:
        {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
            for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_frobenius( out[i], in[i] );
            }
            break;
        }
        }

        dOut.endEdit();

        if(!BlockType::constant && this->assemble.getValue()) this->updateJ();
    }

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        if( !f_geometricStiffness.getValue() ) return;
        if(BlockType::constant) return;

        Data<typename Inherit::InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<typename Inherit::InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<typename Inherit::OutVecDeriv>& childForceData = *mparams->readF(this->toModel);

        helper::WriteAccessor<Data<typename Inherit::InVecDeriv> > parentForce (parentForceData);
        helper::ReadAccessor<Data<typename Inherit::InVecDeriv> > parentDisplacement (parentDisplacementData);
        helper::ReadAccessor<Data<typename Inherit::OutVecDeriv> > childForce (childForceData);

        if(this->assemble.getValue())
        {
            this->K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
        }
        else
        {
            switch( f_method.getValue().getSelectedId() )
            {
            case SMALL:
            {
                break;
            }
            case QR:
            {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
                for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
                {
                    this->jacobian[i].addDForce_qr( parentForce[i], parentDisplacement[i], childForce[i], mparams->kFactor() );
                }
                break;
            }
            case POLAR:
            {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
                for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
                {
                    this->jacobian[i].addDForce_polar( parentForce[i], parentDisplacement[i], childForce[i], mparams->kFactor() );
                }
                break;
            }
            case SVD:
            {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
                for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
                {
                    this->jacobian[i].addDForce_svd( parentForce[i], parentDisplacement[i], childForce[i], mparams->kFactor() );
                }
                break;
            }
            case FROBENIUS:
            {
#ifdef _OPENMP
        #pragma omp parallel for if (this->d_parallel.getValue())
#endif
                for( int i=0 ; i < static_cast<int>(this->jacobian.size()) ; i++ )
                {
                    this->jacobian[i].addDForce_frobenius( parentForce[i], parentDisplacement[i], childForce[i], mparams->kFactor() );
                }
                break;
            }
            }
        }
    }

};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
