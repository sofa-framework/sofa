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
#ifndef SOFA_StabilizedHookeFORCEFIELD_H
#define SOFA_StabilizedHookeFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/StabilizedHookeMaterialBlock.h"


namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply Hooke's Law for isotropic homogeneous incompressible materials from principal stretches.
  * This is the stabilized formulation from "Energetically Consistent Invertible Elasticity", SCA'12
  *
  * The energy is : mu.sum_i((Ui-1)^2) + lambda/2(J-1)^2
  *
  @author Matthieu Nesme

*/

template <class _DataTypes>
class StabilizedHookeForceField : public BaseMaterialForceFieldT<defaulttype::StabilizedHookeMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::StabilizedHookeMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(StabilizedHookeForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulus;
    Data<helper::vector<Real> > _poissonRatio;
    //@}

    virtual void reinit()
    {
        Real youngModulus=0,poissonRatio=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[0];
            if(i<_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[i]; else if(_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[0];

            assert( helper::isClamped<Real>( poissonRatio, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

            this->material[i].init( youngModulus, poissonRatio );
        }
        Inherit::reinit();
    }


protected:
    StabilizedHookeForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,helper::vector<Real>((int)1,(Real)5000),"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,helper::vector<Real>((int)1,(Real)0),"poissonRatio","Poisson Ratio ]-1,0.5["))
//        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
    {
    }

    virtual ~StabilizedHookeForceField()     {    }

};


}
}
}

#endif
