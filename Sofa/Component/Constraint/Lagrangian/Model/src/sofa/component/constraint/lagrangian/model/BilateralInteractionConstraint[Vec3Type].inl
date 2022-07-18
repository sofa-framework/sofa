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
#pragma once

#include <sofa/component/constraint/lagrangian/model/BilateralInteractionConstraint[Vec3Type].h>
#include <sofa/component/constraint/lagrangian/model/BilateralInteractionConstraint.inl>

#include <sofa/type/RGBAColor.h>
#include <sofa/core/ConstraintParams.h>
#include <algorithm> // for std::min

namespace sofa::component::constraint::lagrangian::model::bilateralinteractionconstraint
{

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::init()
{
    unspecializedInit();
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams,
                                                                       BaseVector *v,
                                                                       const DataVecCoord &d_x1, const DataVecCoord &d_x2
                                                                       , const DataVecDeriv & d_v1, const DataVecDeriv & d_v2)
{
    if (!activated) return;

    const type::vector<int> &m1Indices = m1.getValue();
    const type::vector<int> &m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());

    const VecDeriv& restVector = this->restVector.getValue();

    if (cParams->constOrder() == ConstraintParams::VEL)
    {
        getVelocityViolation(v, d_x1, d_x2, d_v1, d_v2);
        return;
    }

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

    if (!merge.getValue())
    {
        dfree.resize(minp);

        for (unsigned pid=0; pid<minp; pid++)
        {
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
                dfree[pid] -= restVector[pid];

            v->set(cid[pid]  , dfree[pid][0]);
            v->set(cid[pid]+1, dfree[pid][1]);
            v->set(cid[pid]+2, dfree[pid][2]);
        }
    }
    else
    {
        for (unsigned pid=0; pid<minp; pid++)
        {
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
                dfree[pid] -= restVector[pid];

            for (unsigned int i=0; i<3; i++)
            {
                if(squareXYZ[i])
                    v->add(cid[pid]+i  , dfree[pid][i]*dfree[pid][i]);
                else
                {

                    v->add(cid[pid]+i  , dfree[pid][i]*sofa::helper::sign(dfree[pid][i] ) );
                }
            }

        }
    }
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getVelocityViolation(BaseVector *v,
                                                                     const DataVecCoord &d_x1,
                                                                     const DataVecCoord &d_x2,
                                                                     const DataVecDeriv &d_v1,
                                                                     const DataVecDeriv &d_v2)
{
    const type::vector<int> &m1Indices = m1.getValue();
    const type::vector<int> &m2Indices = m2.getValue();

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();
    const VecCoord &v1 = d_v1.getValue();
    const VecCoord &v2 = d_v2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());
    const VecDeriv& restVector = this->restVector.getValue();

    if (!merge.getValue())
    {
        auto pos1 = this->getMState1()->readPositions();
        auto pos2 = this->getMState2()->readPositions();

        const SReal dt = this->getContext()->getDt();
        const SReal invDt = SReal(1.0) / dt;

        for (unsigned pid=0; pid<minp; ++pid)
        {

            Deriv dPos = (pos2[m2Indices[pid]] - pos1[m1Indices[pid]]);
            if (pid < restVector.size())
            {
                dPos -= -restVector[pid];
            }
            dPos *= invDt;
            const Deriv dVfree = v2[m2Indices[pid]] - v1[m1Indices[pid]];

            v->set(cid[pid]  , dVfree[0] + dPos[0] );
            v->set(cid[pid]+1, dVfree[1] + dPos[1] );
            v->set(cid[pid]+2, dVfree[2] + dPos[2] );
        }
    }
    else
    {
        VecDeriv dPrimefree;
        dPrimefree.resize(minp);
        dfree.resize(minp);

        for (unsigned pid=0; pid<minp; pid++)
        {
            dPrimefree[pid] = v2[m2Indices[pid]] - v1[m1Indices[pid]];
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
            {
                dPrimefree[pid] -= restVector[pid];
                dfree[pid] -= restVector[pid];
            }

            std::cout<<" x2 : "<<x2[m2Indices[pid]]<<" - x1 :"<<x1[m1Indices[pid]]<<" = "<<dfree[pid]<<std::endl;
            std::cout<<" v2 : "<<v2[m2Indices[pid]]<<" - v1 :"<<v1[m1Indices[pid]]<<" = "<<dPrimefree[pid]<<std::endl;

            for (unsigned int i=0; i<3; i++)
            {
                if(squareXYZ[i])
                {
                    v->add(cid[pid]+i  , 2*dPrimefree[pid][i]*dfree[pid][i]);
                }
                else
                {
                    v->add(cid[pid]+i  , dPrimefree[pid][i]*sofa::helper::sign(dfree[pid][i] ) );
                }
            }

        }
    }
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q,
                                                           Real /*contactDistance*/, int m1, int m2,
                                                           Coord /*Pfree*/, Coord /*Qfree*/,
                                                           long /*id*/, PersistentID /*localid*/)
{
    WriteAccessor<Data<type::vector<int> > > wm1 = this->m1;
    WriteAccessor<Data<type::vector<int> > > wm2 = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);
    wrest.push_back(Q-P);
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv norm, Coord P, Coord Q, Real
                                                           contactDistance, int m1, int m2,
                                                           long id, PersistentID localid)
{
   addContact(norm, P, Q, contactDistance, m1, m2,
               this->getMState2()->read(ConstVecCoordId::freePosition())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::freePosition())->getValue()[m1],
               id, localid);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv norm, Real contactDistance,
                                                           int m1, int m2, long id, PersistentID localid)
{
    addContact(norm,
               this->getMState2()->read(ConstVecCoordId::position())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::position())->getValue()[m1],
               contactDistance, m1, m2,
               this->getMState2()->read(ConstVecCoordId::freePosition())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::freePosition())->getValue()[m1],
               id, localid);
}


} //namespace sofa::component::constraint::lagrangian::model::bilateralinteractionconstraint
