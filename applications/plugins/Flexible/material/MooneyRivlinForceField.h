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
#ifndef SOFA_MooneyRivlinFORCEFIELD_H
#define SOFA_MooneyRivlinFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/MooneyRivlinMaterialBlock.h"

namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply MooneyRivlin's Law for isotropic homogeneous incompressible materials.
  * The energy is : C1 ( I1/ J^2/3  - 3)  + C2 ( I2/ J^4/3  - 3) + bulk/2 (J-1)^2
*/

template <class _DataTypes>
class MooneyRivlinForceField : public BaseMaterialForceFieldT<defaulttype::MooneyRivlinMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::MooneyRivlinMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(MooneyRivlinForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > f_C1;
    Data<helper::vector<Real> > f_C2;
    Data<helper::vector<Real> > f_bulk;
    Data<bool > f_PSDStabilization;
    //@}

    virtual void reinit()
    {
        Real C1=0,C2=0,bulk=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<f_C1.getValue().size()) C1=f_C1.getValue()[i]; else if(f_C1.getValue().size()) C1=f_C1.getValue()[0];
            if(i<f_C2.getValue().size()) C2=f_C2.getValue()[i]; else if(f_C2.getValue().size()) C2=f_C2.getValue()[0];
            if(i<f_bulk.getValue().size()) bulk=f_bulk.getValue()[i]; else if(f_bulk.getValue().size()) bulk=f_bulk.getValue()[0];
            this->material[i].init( C1, C2, bulk, f_PSDStabilization.getValue() );
        }
        Inherit::reinit();
    }



protected:
    MooneyRivlinForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , f_C1(initData(&f_C1,helper::vector<Real>((int)1,(Real)1000),"C1","weight of (~I1-3) term in energy"))
        , f_C2(initData(&f_C2,helper::vector<Real>((int)1,(Real)1000),"C2","weight of (~I2-3) term in energy"))
        , f_bulk(initData(&f_bulk,helper::vector<Real>((int)1,(Real)0),"bulk","bulk modulus (working on I3=J=detF=volume variation)"))
        , f_PSDStabilization(initData(&f_PSDStabilization,false,"PSDStabilization","project stiffness matrix to its nearest symmetric, positive semi-definite matrix"))
//        , _viscosity(initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
    {
    }

    virtual ~MooneyRivlinForceField()     {    }

};


}
}
}

#endif
