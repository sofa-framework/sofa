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
#ifndef SOFA_COMPONENT_MAPPING_RelativeStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_RelativeStrainMAPPING_H

#include <Flexible/config.h>
#include "BaseStrainMapping.h"
#include "RelativeStrainJacobianBlock.h"

#include <sofa/helper/OptionsGroup.h>


namespace sofa
{
namespace component
{
namespace mapping
{

/// Decompose the total strain to an elastic strain - an offset
template <class TStrain>
class RelativeStrainMapping : public BaseStrainMappingT<defaulttype::RelativeStrainJacobianBlock<TStrain> >
{
public:

    typedef defaulttype::RelativeStrainJacobianBlock<TStrain> BlockType;
    typedef BaseStrainMappingT<BlockType> Inherit;
    typedef typename Inherit::Real Real;

    SOFA_CLASS(SOFA_TEMPLATE(RelativeStrainMapping,TStrain), SOFA_TEMPLATE(BaseStrainMappingT,BlockType));


    /// @name  Different ways to decompose the strain
    //@{
    //    enum RelativeMethod { ADDITION=0, MULTIPLICATION, NB_PlasticMethod }; ///< ADDITION -> Müller method (faster), MULTIPLICATION -> Fedkiw method
    //    Data<helper::OptionsGroup> f_method;
    //@}

    /// @name  Strain offset
    //@{
    Data<typename Inherit::InVecCoord> offset;
    Data<bool> inverted;
    //@}


    virtual void reinit()
    {
        typename Inherit::InCoord off = typename Inherit::InCoord();
        if(offset.getValue().size()==1) off = offset.getValue()[0];

        for( size_t i=0 ; i<this->jacobian.size() ; i++ )
        {
            if(i<offset.getValue().size()) off = offset.getValue()[i];
            this->jacobian[i].init(off,inverted.getValue());
        }

        Inherit::reinit();
    }

protected:

    RelativeStrainMapping( core::State<TStrain>* from = NULL, core::State<TStrain>* to = NULL )
        : Inherit ( from, to )
        //        , f_method ( initData ( &f_method,"method","" ) )
        , offset(initData(&offset,"offset","Strain offset"))
        , inverted( initData(&inverted, false, "inverted", "offset-Strain (rather than Strain-offset )") )
    {
        //        helper::OptionsGroup Options;
        //        Options.setNbItems( NB_PlasticMethod );
        //        Options.setItemName( ADDITION,       "addition" );
        //        Options.setItemName( MULTIPLICATION, "multiplication" );
        //        Options.setSelectedItem( ADDITION );
        //        f_method.setValue( Options );
    }

    virtual ~RelativeStrainMapping() { }

    //    virtual void apply( const core::MechanicalParams * /*mparams*/ , Data<typename Inherit::OutVecCoord>& dOut, const Data<typename Inherit::InVecCoord>& dIn )
    //    {
    //        helper::ReadAccessor<Data<typename Inherit::InVecCoord> > inpos (*this->fromModel->read(core::ConstVecCoordId::position()));
    //        helper::ReadAccessor<Data<typename Inherit::OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
    //        if(inpos.size()!=outpos.size()) this->resizeOut();

    //        typename Inherit::OutVecCoord& out = *dOut.beginWriteOnly();
    //        const typename Inherit::InVecCoord&  in  =  dIn.getValue();

    //        typename Inherit::InCoord off = typename Inherit::InCoord();
    //        if(offset.getValue().size()==1) off = offset.getValue()[0];

    //        switch( f_method.getValue().getSelectedId() )
    //        {
    //        case MULTIPLICATION:
    //        {
    //            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
    //            {
    //                out[i] = typename Inherit::OutCoord();
    //                if(i<offset.getValue().size()) off = offset.getValue()[i];
    //                this->jacobian[i].addapply_multiplication( out[i], in[i], off );
    //            }
    //            break;
    //        }
    //        case ADDITION:
    //        {
    //            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
    //            {
    //                out[i] = typename Inherit::OutCoord();
    //                if(i<offset.getValue().size()) off = offset.getValue()[i];
    //                this->jacobian[i].addapply_addition( out[i], in[i],  off );
    //            }
    //            break;
    //        }
    //        }

    //        dOut.endEdit();
    //    }

}; // class RelativeStrainMapping


} // namespace mapping
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_RelativeStrainMAPPING_H
