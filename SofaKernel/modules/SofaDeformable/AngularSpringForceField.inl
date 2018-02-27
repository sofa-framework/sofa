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
#ifndef SOFA_COMPONENT_FORCEFIELD_ANGULARSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_ANGULARSPRINGFORCEFIELD_INL

#include "AngularSpringForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <assert.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
AngularSpringForceField<DataTypes>::AngularSpringForceField()
    : indices(initData(&indices, "indices", "index of nodes controlled by the angular springs"))
    , angularStiffness(initData(&angularStiffness, "angularStiffness", "angular stiffness for the controlled nodes"))
    , angularLimit(initData(&angularLimit, "limit", "angular limit (max; min) values where the force applies"))
    , drawSpring(initData(&drawSpring,false,"drawSpring","draw Spring"))
    , springColor(initData(&springColor,sofa::defaulttype::Vec4f(0.0,1.0,0.0,1.0), "springColor","spring color"))
{    
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::bwdInit()
{
    core::behavior::ForceField<DataTypes>::init();

    if (angularStiffness.getValue().empty())
    {
		msg_info("AngularSpringForceField") << "No angular stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << "\n";

        VecReal stiffs;
        stiffs.push_back(100.0);
        angularStiffness.setValue(stiffs);
    }
    this->k = angularStiffness.getValue();

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (this->getContext()->getMechanicalState());
    if (!mState) {
		msg_error("AngularSpringForceField") << "MechanicalStateFilter has no binding MechanicalState" << "\n";
    }
    matS.resize(mState->getMatrixSize(), mState->getMatrixSize());
}

template<class DataTypes>
void AngularSpringForceField<DataTypes>::reinit()
{
    if (angularStiffness.getValue().empty())
    {
		msg_info("AngularSpringForceField") << "nN angular stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << "\n";

        VecReal stiffs;
        stiffs.push_back(100.0);
        angularStiffness.setValue(stiffs);
    }
    this->k = angularStiffness.getValue();
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    if(!mState) {
		msg_info("AngularSpringForceField") << "No Mechanical State found, no force will be computed..." << "\n";
        return;
    }

    sofa::helper::WriteAccessor< DataVecDeriv > f1 = f;
    sofa::helper::ReadAccessor< DataVecCoord > p1 = x;
    f1.resize(p1.size());
    for (unsigned int i = 1; i < indices.getValue().size(); i++)
    {
        const unsigned int index = indices.getValue()[i];
        defaulttype::Quat dq = p1[index].getOrientation() * p1[index-1].getOrientation().inverse();
        defaulttype::Vec3d axis;
        double angle = 0.0;
        Real stiffness;
        dq.normalize();

        Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

		// to avoid numerical instabilities of acos for theta < 5
		if(dq[3]>0.999999) // theta < 5 -> q[3] = cos(theta/2) > 0.999
        {
            sin_half_theta = sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2]);
            angle = (Real)(2.0 * asin(sin_half_theta));
        }
        else
        {
            Real half_theta = acos(dq[3]);
            sin_half_theta = sin(half_theta);
            angle = 2*half_theta;
        }

        assert(sin_half_theta>=0);
        if (sin_half_theta < std::numeric_limits<Real>::epsilon())
            axis = defaulttype::Vec<3,Real>(0.0, 1.0, 0.0);
        else
            axis = defaulttype::Vec<3,Real>(dq[0], dq[1], dq[2])/sin_half_theta;

		if (i < this->k.size())
            stiffness = this->k[i] = angularStiffness.getValue()[i];
         else
            stiffness = this->k[0] = angularStiffness.getValue()[0];

        getVOrientation(f1[index]) -= axis * angle * stiffness;
    }
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< DataVecDeriv > df1 = df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx1 = dx;

    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    for (unsigned int i=0; i<indices.getValue().size(); i++)
        getVOrientation(df1[indices.getValue()[i]]) -=  getVOrientation(dx1[indices.getValue()[i]]) * (i < this->k.size() ? this->k[i] : this->k[0]) * kFactor ;
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const int N = 6;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int curIndex = 0;
    for (unsigned int index = 0; index < indices.getValue().size(); index++)
    {
//        if (angle <  (angularLimit.getValue()[2*i]/180.0*M_PI) || angle >  (angularLimit.getValue()[2*i+1]/180.0*M_PI))  {
            curIndex = indices.getValue()[index];
            for(int i = 3; i < 6; i++)
                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * (index < this->k.size() ? this->k[index] : this->k[0]));
//        }
    }
}

template<class DataTypes>
void AngularSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !drawSpring.getValue())
        return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    sofa::helper::ReadAccessor< DataVecCoord > p = this->mstate->read(core::VecCoordId::position());
    sofa::helper::vector< defaulttype::Vec3d > vertices;

    for (unsigned int i=0; i<indices.getValue().size(); i++)
    {
        const unsigned int index = indices.getValue()[i];
        vertices.push_back(p[index].getCenter());
    }
    vparams->drawTool()->drawLines(vertices,5,springColor.getValue());
    vparams->drawTool()->restoreLastState();
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_ANGULARSPRINGFORCEFIELD_INL



