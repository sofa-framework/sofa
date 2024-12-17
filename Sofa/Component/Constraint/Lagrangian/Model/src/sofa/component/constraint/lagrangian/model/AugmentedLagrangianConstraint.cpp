/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_CONSTRAINTSET_AugmentedLagrangianConstraint_CPP
#include <sofa/component/constraint/lagrangian/model/BaseContactLagrangianConstraint.inl>
#include <sofa/component/constraint/lagrangian/model/AugmentedLagrangianConstraint.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::lagrangian::model
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

int AugmentedLagrangianConstraintClass = core::RegisterObject("AugmentedLagrangianConstraint")
        .add< AugmentedLagrangianConstraint<Vec3Types> >()

        ;


template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BaseContactLagrangianConstraint<Vec3Types,AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API AugmentedLagrangianConstraint<Vec3Types>;



void AugmentedLagrangianResolutionWithFriction::init(int line, SReal** w, SReal* force)
{
    _W[0]=w[line  ][line  ];
    _W[1]=w[line  ][line+1];
    _W[2]=w[line  ][line+2];
    _W[3]=w[line+1][line+1];
    _W[4]=w[line+1][line+2];
    _W[5]=w[line+2][line+2];

    ////////////////// christian : the following does not work ! /////////
    if(_prev)
    {
        force[line] = _prev->popForce();
        force[line+1] = _prev->popForce();
        force[line+2] = _prev->popForce();
    }

}

void AugmentedLagrangianResolutionWithFriction::resolution(int line, SReal** /*w*/, SReal* d, SReal* force, SReal * /*dfree*/)
{
    force[line] -= d[line] * _epsilon;

    if(force[line] < 0)
    {
        force[line]=0; force[line+1]=0; force[line+2]=0;
        return;
    }

    const SReal f_t_0 = force[line + 1] - d[line+ 1] * _epsilon;
    const SReal f_t_1 = force[line + 2] - d[line+ 2] * _epsilon;

    const SReal criteria = sqrt(pow(f_t_0,2.0) + pow(f_t_1,2.0)) - _mu * fabs(force[line]);

    if(criteria<0)
    {
        force[line+1] = f_t_0 ;
        force[line+2] = f_t_1 ;
    }
    else
    {
        const SReal norm_s = sqrt(pow(d[line+ 1],2.0) + pow(d[line+ 2],2.0));
        force[line+1] -= _mu * d[line] * _epsilon * d[line+ 1]/norm_s;
        force[line+2] -= _mu * d[line] * _epsilon * d[line+ 2]/norm_s;
    }
}

void AugmentedLagrangianResolutionWithFriction::store(int line, SReal* force, bool /*convergence*/)
{
    if(_prev)
    {
        _prev->pushForce(force[line]);
        _prev->pushForce(force[line+1]);
        _prev->pushForce(force[line+2]);
    }

    if(_active)
    {
        *_active = (force[line] != 0);
        _active = nullptr; // Won't be used in the haptic thread
    }
}


} //namespace sofa::component::constraint::lagrangian::model
