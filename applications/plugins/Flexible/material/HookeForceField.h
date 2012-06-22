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
//#include "../material/HookeMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>



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

            this->material[i].init( youngModulus, poissonRatio, viscosity );
        }
        Inherit::reinit();
    }

    /// Set the constraint value
    virtual void writeConstraintValue(const core::MechanicalParams* params, core::MultiVecDerivId constraintId )
    {
        if( ! this->isCompliance.getValue() ) return; // if not seen as a compliance, then apply  forces in addForce

        helper::ReadAccessor< typename Inherit::DataVecCoord > x = params->readX(this->mstate);
        helper::ReadAccessor< typename Inherit::DataVecDeriv > v = params->readV(this->mstate);
        helper::WriteAccessor<typename Inherit::DataVecDeriv > c = *constraintId[this->mstate.get(params)].write();
        Real alpha = params->implicitVelocity();
        Real beta  = params->implicitPosition();
        Real h     = params->dt();
        Real d     = this->getDampingRatio();

        for(unsigned i=0; i<c.size(); i++)
            c[i] = -( x[i] + v[i] * (d + alpha*h) ) * (1./ (alpha * (h*beta +d)));
    }

    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulus.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    HookeForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,vector<Real>((int)1,(Real)5000),"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,vector<Real>((int)1,(Real)0.45),"poissonRatio","Poisson Ratio"))
        , _viscosity(initData(&_viscosity,vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
        // _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeForceField()     {    }


};


}
}
}

#endif
