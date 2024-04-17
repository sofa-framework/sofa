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
#define SOFA_COMPONENT_CONSTRAINTSET_UNILATERALLAGRANGIANCONSTRAINT_CPP
#include <sofa/component/constraint/lagrangian/model/UnilateralLagrangianConstraint.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::lagrangian::model
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

//TODO(dmarchal) What does this TODO mean ?
int UnilateralLagrangianConstraintClass = core::RegisterObject("TODO-UnilateralLagrangianConstraint")
        .add< UnilateralLagrangianConstraint<Vec3Types> >()

        ;


template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API UnilateralLagrangianConstraint<Vec3Types>;



void UnilateralConstraintResolutionWithFriction::init(int line, SReal** w, SReal* force)
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

void UnilateralConstraintResolutionWithFriction::resolution(int line, SReal** /*w*/, SReal* d, SReal* force, SReal * /*dfree*/)
{
    SReal f[2];
    SReal normFt;

    f[0] = force[line]; f[1] = force[line+1];
    force[line] -= d[line] / _W[0];

    if(force[line] < 0)
    {
        force[line]=0; force[line+1]=0; force[line+2]=0;
        return;
    }

    d[line+1] += _W[1] * (force[line]-f[0]);
    d[line+2] += _W[2] * (force[line]-f[0]);
    force[line+1] -= 2*d[line+1] / (_W[3] +_W[5]) ;
    force[line+2] -= 2*d[line+2] / (_W[3] +_W[5]) ;

    normFt = sqrt(force[line+1]*force[line+1] + force[line+2]*force[line+2]);

    const SReal fN = _mu*force[line];
    if(normFt > fN)
    {
        const SReal factor = fN / normFt;
        force[line+1] *= factor;
        force[line+2] *= factor;
    }
}

void UnilateralConstraintResolutionWithFriction::store(int line, SReal* force, bool /*convergence*/)
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
