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
#ifndef SOFA_ProjectiveFORCEFIELD_H
#define SOFA_ProjectiveFORCEFIELD_H

#include <Flexible/config.h>
#include "../material/BaseMaterialForceField.h"
#include "../material/ProjectiveMaterialBlock.h"


namespace sofa
{
namespace component
{
namespace forcefield
{


/**
    Forcefield based on a quadratic distance measure between dofs and a projection
    cf. paper 'projective dynamics' siggraph 14.
    Stiffness is consant, thus preinversible.
    Projecting deformation gradients to rotations to equivalent to corotational Hooke law (without Poisson ratio, nor geometric stiffness)
*/

template <class _DataTypes>
class ProjectiveForceField : public BaseMaterialForceFieldT<defaulttype::ProjectiveMaterialBlock<_DataTypes> >
{
public:
    typedef defaulttype::ProjectiveMaterialBlock<_DataTypes> BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(ProjectiveForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulus;
    Data<helper::vector<Real> > _viscosity;
    //@}

    virtual void reinit()
    {
        Real youngModulus=0,viscosity=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[0];
            if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

            this->material[i].init( youngModulus, viscosity );
        }
        Inherit::reinit();
    }


    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulus.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    ProjectiveForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,helper::vector<Real>((int)1,(Real)5000),"youngModulus","Young Modulus"))
        , _viscosity(initData(&_viscosity,helper::vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
    }

    virtual ~ProjectiveForceField()     {    }
};



}
}
}

#endif
