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
#ifndef SOFA_VolumePreservationFORCEFIELD_H
#define SOFA_VolumePreservationFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/VolumePreservationMaterialBlock.inl"

#include <sofa/helper/OptionsGroup.h>

namespace sofa
{
namespace component
{
namespace forcefield
{
;

/** Apply VolumePreservation's Law for isotropic homogeneous incompressible materials.
  * The energy is : k/2 ln( I3^1/2 )^2 = k/2 ln( (U1*U2*U3)^1/2 )^2
*/

template <class _DataTypes>
class VolumePreservationForceField : public BaseMaterialForceFieldT<defaulttype::VolumePreservationMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::VolumePreservationMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(VolumePreservationForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::OptionsGroup> f_method; ///< energy form
    Data<helper::vector<Real> > f_k; ///< bulk modulus: weight ln(J)^2/2 term in energy 
    //@}

    virtual void reinit()
    {
        Real k=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<f_k.getValue().size()) k=f_k.getValue()[i]; else if(f_k.getValue().size()) k=f_k.getValue()[0];
            this->material[i].init( k );
        }
        Inherit::reinit();
    }


    virtual SReal getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const typename Inherit::DataVecCoord& x ) const
    {
        SReal e = 0;
        const typename Inherit::VecCoord& _x = x.getValue();

        switch( f_method.getValue().getSelectedId() )
        {
            case 0:
            {
                for( unsigned int i=0 ; i<this->material.size() ; i++ )
                {
                    e += this->material[i].getPotentialEnergy_method0( _x[i] );
                }
                break;
            }
            case 1:
            {
                for( unsigned int i=0 ; i<this->material.size() ; i++ )
                {
                    e += this->material[i].getPotentialEnergy_method1( _x[i] );
                }
                break;
            }
        }
        return e;
    }

protected:
    VolumePreservationForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , f_method ( initData ( &f_method,"method","energy form" ) )
        , f_k(initData(&f_k,helper::vector<Real>((int)1,(Real)0),"k","bulk modulus: weight ln(J)^2/2 term in energy "))
    {
        helper::OptionsGroup Options(2	,"0 - k.ln(J)^2/2"
                ,"1 - k.(J-1)^2/2" );
        Options.setSelectedItem(1);
        f_method.setValue(Options);
    }

    virtual ~VolumePreservationForceField() {}



    virtual void addForce(const core::MechanicalParams* /*mparams*/, typename Inherit::DataVecDeriv& _f , const typename Inherit::DataVecCoord& _x , const typename Inherit::DataVecDeriv& _v)
    {
        typename Inherit::VecDeriv&  f = *_f.beginEdit();
        const typename Inherit::VecCoord&  x = _x.getValue();
        const typename Inherit::VecDeriv&  v = _v.getValue();

        switch( f_method.getValue().getSelectedId() )
        {
        case 0:
        {
            for( unsigned int i=0 ; i<this->material.size() ; i++ )
            {
                this->material[i].addForce_method0(f[i],x[i],v[i]);
            }
            break;
        }
        case 1:
        {
            for( unsigned int i=0 ; i<this->material.size() ; i++ )
            {
                this->material[i].addForce_method1(f[i],x[i],v[i]);
            }
            break;
        }
        }

        _f.endEdit();

        if( !BlockType::constantK && this->assemble.getValue() )
        {
            this->updateC();
            this->updateK();
            this->updateB();
        }
    }


};


}
}
}

#endif
