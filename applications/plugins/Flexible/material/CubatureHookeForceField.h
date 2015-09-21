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
#ifndef SOFA_CUBATUREHookeFORCEFIELD_H
#define SOFA_CUBATUREHookeFORCEFIELD_H

#include "../initFlexible.h"
#include "../material/CubatureBaseMaterialForceField.h"
#include "../material/HookeMaterialBlock.inl"
//#include "../material/HookeMaterialBlock.h"
//#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>


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
class CubatureHookeForceField : public CubatureBaseMaterialForceFieldT<defaulttype::HookeMaterialBlock<_DataTypes, defaulttype::IsotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> > >
{
public:
    typedef defaulttype::IsotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock<_DataTypes, LawType > BlockType;
    typedef CubatureBaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(CubatureHookeForceField,_DataTypes),SOFA_TEMPLATE(CubatureBaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<vector<Real> > _youngModulus;
    Data<vector<Real> > _poissonRatio;
    Data<vector<Real> > _viscosity;
    //@}

    virtual void reinit()
    {
        Real youngModulus=0,poissonRatio=0,viscosity=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[0];
            if(i<_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[i]; else if(_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[0];
            if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

            assert( helper::isClamped<Real>( poissonRatio, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

            std::vector<Real> params; params.push_back(youngModulus); params.push_back(poissonRatio);
            this->material[i].init( params, viscosity );
        }
        Inherit::reinit();
    }


    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            if(_youngModulus.isDirty() || _poissonRatio.isDirty() || _viscosity.isDirty()) reinit();
        }
    }


    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulus.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    CubatureHookeForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,vector<Real>((int)1,(Real)5000),"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,vector<Real>((int)1,(Real)0),"poissonRatio","Poisson Ratio ]-1,0.5["))
        , _viscosity(initData(&_viscosity,vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
        // _poissonRatio.setWidget("poissonRatio");
        this->f_listening.setValue(true);
    }

    virtual ~CubatureHookeForceField()     {    }
};
}
}
}

#endif
