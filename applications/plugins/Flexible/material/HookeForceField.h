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
#ifndef SOFA_HookeFORCEFIELD_H
#define SOFA_HookeFORCEFIELD_H

#include "../initFlexible.h"
#include "../material/BaseMaterialForceField.h"
#include "../material/HookeMaterialBlock.inl"

namespace sofa
{
namespace component
{
namespace forcefield
{

using helper::vector;

/** Compute stress from strain (=apply material law)
  * using Hooke's Law for isotropic homogeneous materials:
*/

template <class _DataTypes>
class SOFA_Flexible_API HookeForceField : public BaseMaterialForceField<defaulttype::HookeMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::HookeMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceField<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(HookeForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceField, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<Real> _youngModulus;
    Data<Real> _poissonRatio;
    Data<Real> _viscosity;
    //@}

    virtual void reinit()
    {
        for(unsigned int i=0; i<this->material.size(); i++) this->material[i].init(this->_youngModulus.getValue(),this->_poissonRatio.getValue(),this->_viscosity.getValue());
        Inherit::reinit();
    }

protected:
    HookeForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,(Real)0.45f,"poissonRatio","Poisson Ratio"))
        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
    {
        _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeForceField()     {    }

};


}
}
}

#endif
