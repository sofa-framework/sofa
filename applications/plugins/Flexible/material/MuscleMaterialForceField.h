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
#ifndef SOFA_MuscleMaterialFORCEFIELD_H
#define SOFA_MuscleMaterialFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/MuscleMaterialBlock.h"


namespace sofa
{
namespace component
{
namespace forcefield
{


/** Apply exponential law for active muscles
*/

template <class _DataTypes>
class MuscleMaterialForceField : public BaseMaterialForceFieldT<defaulttype::MuscleMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::MuscleMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(MuscleMaterialForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > f_lambda0;
    Data<helper::vector<Real> > f_sigmaMax;
    Data<helper::vector<Real> > f_a;
    Data<helper::vector<Real> > f_b;
    Data<helper::vector<Real> > f_Vvm;
    Data<helper::vector<Real> > f_Ver;
    Data<helper::vector<Real> > f_Vsh;
    //@}

    virtual void reinit()
    {
        Real b=0,Vvm=0,Ver=0,lambda0=0,Vsh=0,a=0,sigmaMax=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<f_lambda0.getValue().size()) lambda0=f_lambda0.getValue()[i]; else if(f_lambda0.getValue().size()) lambda0=f_lambda0.getValue()[0];
            if(i<f_sigmaMax.getValue().size()) sigmaMax=f_sigmaMax.getValue()[i]; else if(f_sigmaMax.getValue().size()) sigmaMax=f_sigmaMax.getValue()[0];
            if(i<f_a.getValue().size()) a=f_a.getValue()[i]; else if(f_a.getValue().size()) a=f_a.getValue()[0];
            if(i<f_b.getValue().size()) b=f_b.getValue()[i]; else if(f_b.getValue().size()) b=f_b.getValue()[0];
            if(i<f_Vvm.getValue().size()) Vvm=f_Vvm.getValue()[i]; else if(f_Vvm.getValue().size()) Vvm=f_Vvm.getValue()[0];
            if(i<f_Ver.getValue().size()) Ver=f_Ver.getValue()[i]; else if(f_Ver.getValue().size()) Ver=f_Ver.getValue()[0];
            if(i<f_Vsh.getValue().size()) Vsh=f_Vsh.getValue()[i]; else if(f_Vsh.getValue().size()) Vsh=f_Vsh.getValue()[0];
            this->material[i].init( lambda0,sigmaMax,a,b,Vvm,Ver,Vsh);
        }
        Inherit::reinit();
    }


protected:
    MuscleMaterialForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , f_lambda0(initData(&f_lambda0,helper::vector<Real>((int)1,(Real)1.),"lambda0","optimal fiber stretch"))
        , f_sigmaMax(initData(&f_sigmaMax,helper::vector<Real>((int)1,(Real)3E5),"sigmaMax","maximum isometric stress"))
        , f_a(initData(&f_a,helper::vector<Real>((int)1,(Real)0),"a","activation level"))
        , f_b(initData(&f_b,helper::vector<Real>((int)1,(Real)0.5),"b",""))
        , f_Vvm(initData(&f_Vvm,helper::vector<Real>((int)1,(Real)10),"Vvm",""))
        , f_Ver(initData(&f_Ver,helper::vector<Real>((int)1,(Real)0.5),"Ver",""))
        , f_Vsh(initData(&f_Vsh,helper::vector<Real>((int)1,(Real)0.3),"Vsh",""))
    {
    }

    virtual ~MuscleMaterialForceField()     {    }

};


}
}
}

#endif
