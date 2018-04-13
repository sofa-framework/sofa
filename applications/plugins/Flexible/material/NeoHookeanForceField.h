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
#ifndef SOFA_NeoHookeanFORCEFIELD_H
#define SOFA_NeoHookeanFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/NeoHookeanMaterialBlock.h"


namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply NeoHookean's Law for isotropic homogeneous incompressible materials.
  * The energy is : mu/2 ( I1/ I3^1/3  - 3) + bulk/2 (I3-1)^2
*/

template <class _DataTypes>
class NeoHookeanForceField : public BaseMaterialForceFieldT<defaulttype::NeoHookeanMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::NeoHookeanMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(NeoHookeanForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulus; ///< stiffness
    Data<helper::vector<Real> > _poissonRatio; ///< incompressibility ]-1,0.5[
    Data<bool > f_PSDStabilization; ///< project stiffness matrix to its nearest symmetric, positive semi-definite matrix
    //@}

    virtual void reinit()
    {
        Real ym=0,pr=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) ym=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) ym=_youngModulus.getValue()[0];
            if(i<_poissonRatio.getValue().size()) pr=_poissonRatio.getValue()[i]; else if(_poissonRatio.getValue().size()) pr=_poissonRatio.getValue()[0];

            assert( helper::isClamped<Real>( pr, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );
            assert( helper::isClamped( pr, -1+std::numeric_limits<Real>::epsilon(), (Real)0.5-std::numeric_limits<Real>::epsilon() ) );

            this->material[i].init( ym, pr, f_PSDStabilization.getValue() );
        }
        Inherit::reinit();
    }


protected:
    NeoHookeanForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,helper::vector<Real>((int)1,(Real)1000),"youngModulus","stiffness"))
        , _poissonRatio(initData(&_poissonRatio,helper::vector<Real>((int)1,(Real)0),"poissonRatio","incompressibility ]-1,0.5["))
        , f_PSDStabilization(initData(&f_PSDStabilization,false,"PSDStabilization","project stiffness matrix to its nearest symmetric, positive semi-definite matrix"))
//        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
    {
    }

    virtual ~NeoHookeanForceField()     {    }

};


}
}
}

#endif
