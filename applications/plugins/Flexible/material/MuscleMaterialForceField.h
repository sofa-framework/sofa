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
#ifndef SOFA_MuscleMaterialFORCEFIELD_H
#define SOFA_MuscleMaterialFORCEFIELD_H

#include "../initFlexible.h"
#include "../material/BaseMaterialForceField.h"
#include "../material/MuscleMaterialBlock.h"

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using helper::vector;

/** Apply Blemker Material's Law for muscle materials.
    from 2005 paper: "A 3D model of muscle reveals the causes of nonuniform strains in the biceps brachii"
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
    Data<vector<Real> > f_P1;
    Data<vector<Real> > f_P2;
    Data<vector<Real> > f_lambda0;
    Data<vector<Real> > f_lambdaL;
    Data<vector<Real> > f_a;
    Data<vector<Real> > f_sigmaMax;
    //@}

    virtual void reinit()
    {
        Real P1=0,P2=0,lambda0=0,lambdaL=0,a=0,sigmaMax=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<f_P1.getValue().size()) P1=f_P1.getValue()[i]; else if(f_P1.getValue().size()) P1=f_P1.getValue()[0];
            if(i<f_P2.getValue().size()) P2=f_P2.getValue()[i]; else if(f_P2.getValue().size()) P2=f_P2.getValue()[0];
            if(i<f_lambda0.getValue().size()) lambda0=f_lambda0.getValue()[i]; else if(f_lambda0.getValue().size()) lambda0=f_lambda0.getValue()[0];
            if(i<f_lambdaL.getValue().size()) lambdaL=f_lambdaL.getValue()[i]; else if(f_lambdaL.getValue().size()) lambdaL=f_lambdaL.getValue()[0];
            if(i<f_a.getValue().size()) a=f_a.getValue()[i]; else if(f_a.getValue().size()) a=f_a.getValue()[0];
            if(i<f_sigmaMax.getValue().size()) sigmaMax=f_sigmaMax.getValue()[i]; else if(f_sigmaMax.getValue().size()) sigmaMax=f_sigmaMax.getValue()[0];
            this->material[i].init( P1,P2,lambda0,lambdaL,a,sigmaMax);
        }
        Inherit::reinit();
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            if(f_P1.isDirty() || f_P2.isDirty() || f_lambda0.isDirty() || f_lambdaL.isDirty() || f_a.isDirty() || f_sigmaMax.isDirty()) reinit();
        }
    }


protected:
    MuscleMaterialForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , f_P1(initData(&f_P1,vector<Real>((int)1,(Real)0.05),"P1",""))
        , f_P2(initData(&f_P2,vector<Real>((int)1,(Real)6.6),"P2",""))
        , f_lambda0(initData(&f_lambda0,vector<Real>((int)1,(Real)1.4),"lambda0","optimal fiber stretch"))
        , f_lambdaL(initData(&f_lambdaL,vector<Real>((int)1,(Real)1.4),"lambdaL","stretch above which passive part is linear"))
        , f_a(initData(&f_a,vector<Real>((int)1,(Real)0),"a","activation level"))
        , f_sigmaMax(initData(&f_sigmaMax,vector<Real>((int)1,(Real)3E5),"sigmaMax","maximum isometric stress"))
    {
        this->f_listening.setValue(true);
    }

    virtual ~MuscleMaterialForceField()     {    }

};


}
}
}

#endif
