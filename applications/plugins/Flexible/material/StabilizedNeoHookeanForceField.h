/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_StabilizedNeoHookeanFORCEFIELD_H
#define SOFA_StabilizedNeoHookeanFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/StabilizedNeoHookeanMaterialBlock.h"


namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply stabilized NeoHookean's Law for isotropic homogeneous incompressible materials from principal stretches.
  * This is the stabilized formulation from "Energetically Consistent Invertible Elasticity", SCA'12
  *
  * W = mu/2(I1-3)-mu.ln(J)+lambda/2(ln(J))^2
  *
  * @author Matthieu Nesme
  *
*/

template <class _DataTypes>
class StabilizedNeoHookeanForceField : public BaseMaterialForceFieldT<defaulttype::StabilizedNeoHookeanMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::StabilizedNeoHookeanMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(StabilizedNeoHookeanForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulus;
    Data<helper::vector<Real> > _poissonRatio;
    //@}

    virtual void reinit()
    {
        Real ym=0,pr=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) ym=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) ym=_youngModulus.getValue()[0];
            if(i<_poissonRatio.getValue().size()) pr=_poissonRatio.getValue()[i]; else if(_poissonRatio.getValue().size()) pr=_poissonRatio.getValue()[0];

            assert( helper::isClamped<Real>( pr, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

            this->material[i].init( ym, pr );
        }
        Inherit::reinit();
    }


protected:
    StabilizedNeoHookeanForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,helper::vector<Real>((int)1,(Real)1000),"youngModulus","stiffness"))
        , _poissonRatio(initData(&_poissonRatio,helper::vector<Real>((int)1,(Real)0),"poissonRatio","incompressibility ]-1,0.5["))
//        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
    {
    }

    virtual ~StabilizedNeoHookeanForceField()     {    }

};


}
}
}

#endif
