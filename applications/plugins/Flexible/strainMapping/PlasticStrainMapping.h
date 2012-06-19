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
#ifndef SOFA_COMPONENT_MAPPING_PlasticStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_PlasticStrainMAPPING_H

#include "../initFlexible.h"
#include "BaseStrainMapping.h"
#include "PlasticStrainJacobianBlock.h"

#include <sofa/helper/OptionsGroup.h>


namespace sofa
{
namespace component
{
namespace mapping
{


/// Decompose the total strain to an elastic strain + a plastic strain
template <class TStrain>
class SOFA_Flexible_API PlasticStrainMapping : public BaseStrainMapping<defaulttype::PlasticStrainJacobianBlock<TStrain> >
{
public:

    typedef defaulttype::PlasticStrainJacobianBlock<TStrain> BlockType;
    typedef BaseStrainMapping<BlockType> Inherit;
    typedef typename Inherit::Real Real;


    SOFA_CLASS(SOFA_TEMPLATE(PlasticStrainMapping,TStrain), SOFA_TEMPLATE(BaseStrainMapping,BlockType));


    /// @name  Different ways to decompose the strain
    //@{
    enum PlasticMethod { ADDITION=0, MULTIPLICATION, NB_PlasticMethod }; ///< ADDITION -> Müller method (faster), MULTIPLICATION -> Fedkiw method
    Data<helper::OptionsGroup> f_method;
    //@}


    /// @name  Plasticity parameters such as "Interactive Virtual Materials", Muller & Gross, GI 2004
    //@{
    Data<Real> _max;
    Data<Real> _yield;
    Real _squaredYield;
    Data<Real> _creep; ///< this parameters is different from the article, here it includes the multiplication by dt
    //@}



    virtual void reinit()
    {
        _squaredYield = _yield.getValue() * _yield.getValue();

        Inherit::reinit();
    }

    virtual void reset()
    {
        //serr<<"PlasticStrainMapping::reset"<<sendl;
        Inherit::reset();

        for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
            this->jacobian[i].reset();
    }


protected:

    PlasticStrainMapping( core::State<TStrain>* from = NULL, core::State<TStrain>* to = NULL )
        : Inherit ( from, to )
        , f_method ( initData ( &f_method,"method","" ) )
        , _max(initData(&_max,(Real)0.1f,"max","Plastic Max Threshold (2-norm of the strain)"))
        , _yield(initData(&_yield,(Real)0.0001f,"yield","Plastic Yield Threshold (2-norm of the strain)"))
        , _creep(initData(&_creep,(Real)1.f,"creep","Plastic Creep Factor * dt [0,1]. 1 <-> pure plastic ; <1 <-> visco-plastic (warning depending on dt)"))
    {
        helper::OptionsGroup Options;
        Options.setNbItems( NB_PlasticMethod );
        Options.setItemName( ADDITION,       "addition" );
        Options.setItemName( MULTIPLICATION, "multiplication" );
        Options.setSelectedItem( ADDITION );
        f_method.setValue( Options );
    }

    virtual ~PlasticStrainMapping() { }

    virtual void apply( const core::MechanicalParams */*mparams*/ , Data<typename Inherit::OutVecCoord>& dOut, const Data<typename Inherit::InVecCoord>& dIn )
    {
        helper::ReadAccessor<Data<typename Inherit::InVecCoord> > inpos (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<typename Inherit::OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
        if(inpos.size()!=outpos.size()) this->resizeOut();

        typename Inherit::OutVecCoord& out = *dOut.beginEdit();
        const typename Inherit::InVecCoord&  in  =  dIn.getValue();

        switch( f_method.getValue().getSelectedId() )
        {
        case MULTIPLICATION:
        {
            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_multiplication( out[i], in[i], _max.getValue(), _squaredYield, _creep.getValue() );
            }
            break;
        }
        case ADDITION:
        {
            for( unsigned int i=0 ; i<this->jacobian.size() ; i++ )
            {
                out[i] = typename Inherit::OutCoord();
                this->jacobian[i].addapply_addition( out[i], in[i], _max.getValue(), _squaredYield, _creep.getValue() );
            }
            break;
        }
        }

        dOut.endEdit();

    }

}; // class PlasticStrainMapping


} // namespace mapping
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PlasticStrainMAPPING_H
