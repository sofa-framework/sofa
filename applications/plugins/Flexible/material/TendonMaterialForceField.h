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
#ifndef SOFA_TendonMaterialFORCEFIELD_H
#define SOFA_TendonMaterialFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/TendonMaterialBlock.h"

namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply Blemker Material's Law for Tendon materials.
    from 2005 paper: "A 3D model of Muscle reveals the causes of nonuniform strains in the biceps brachii"
*/

template <class _DataTypes>
class TendonMaterialForceField : public BaseMaterialForceFieldT<defaulttype::TendonMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::TendonMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(TendonMaterialForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > f_L1;
    Data<helper::vector<Real> > f_L2;
    Data<helper::vector<Real> > f_lambdaL; ///< stretch above which behavior becomes linear
    //@}

    virtual void reinit()
    {
        Real L1=0,L2=0,lambdaL=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<f_L1.getValue().size()) L1=f_L1.getValue()[i]; else if(f_L1.getValue().size()) L1=f_L1.getValue()[0];
            if(i<f_L2.getValue().size()) L2=f_L2.getValue()[i]; else if(f_L2.getValue().size()) L2=f_L2.getValue()[0];
            if(i<f_lambdaL.getValue().size()) lambdaL=f_lambdaL.getValue()[i]; else if(f_lambdaL.getValue().size()) lambdaL=f_lambdaL.getValue()[0];
            this->material[i].init( L1,L2,lambdaL);
        }
        Inherit::reinit();
    }



protected:
    TendonMaterialForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , f_L1(initData(&f_L1,helper::vector<Real>((int)1,(Real)2.7E6),"L1",""))
        , f_L2(initData(&f_L2,helper::vector<Real>((int)1,(Real)46.4),"L2",""))
        , f_lambdaL(initData(&f_lambdaL,helper::vector<Real>((int)1,(Real)1.03),"lambdaL","stretch above which behavior becomes linear"))
    {
    }

    virtual ~TendonMaterialForceField()     {    }

};


}
}
}

#endif
