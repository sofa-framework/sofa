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


    /// @name  Strain offset
    //@{
    Data<typename Inherit::InVecCoord> d_offset; ///< Strain offset
    Data<bool> d_inverted; ///< offset-Strain (rather than Strain-offset )
    //@}

    virtual void reinit()
    {
        bool inverted = d_inverted.getValue();
        for( size_t i=0 ; i<this->jacobian.size() ; i++ )
            this->jacobian[i].init(inverted);

        Inherit::reinit();
    }

protected:

    RelativeStrainMapping( core::State<TStrain>* from = NULL, core::State<TStrain>* to = NULL )
        : Inherit ( from, to )
        , d_offset(initData(&d_offset,"offset","Strain offset"))
        , d_inverted( initData(&d_inverted, false, "inverted", "offset-Strain (rather than Strain-offset )") )
    {
    }

    virtual ~RelativeStrainMapping() { }

    virtual void apply( const core::MechanicalParams * /*mparams*/ , Data<typename Inherit::OutVecCoord>& dOut, const Data<typename Inherit::InVecCoord>& dIn )
    {
        helper::ReadAccessor<Data<typename Inherit::InVecCoord> > inpos (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<typename Inherit::OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<typename Inherit::InVecCoord> > offset (this->d_offset);
        if(inpos.size()!=outpos.size()) this->resizeOut();

        typename Inherit::OutVecCoord& out = *dOut.beginWriteOnly();
        const typename Inherit::InVecCoord&  in  =  dIn.getValue();

        if(offset.size()==0)
        {
            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
                out[i] =in[i];
        }
        else
            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_diff( out[i], in[i], offset[ std::min((unsigned int)offset.size()-1,i) ] );
            }
        dOut.endEdit();
    }

}; // class RelativeStrainMapping


} // namespace mapping
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_RelativeStrainMAPPING_H
