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
#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
#include <sofa/core/ConstraintParams.h>
#include <algorithm> // for std::min

namespace sofa::component::constraint::lagrangian::model
{

using sofa::core::objectmodel::KeypressedEvent ;
using sofa::core::objectmodel::Event ;
using sofa::helper::WriteAccessor ;
using sofa::type::Vec;

template<class DataTypes>
BilateralLagrangianConstraint<DataTypes>::BilateralLagrangianConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , m1(initData(&m1, "first_point","index of the constraint on the first model"))
    , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
    , d_numericalTolerance(initData(&d_numericalTolerance, 0.0001, "numericalTolerance",
                                    "a real value specifying the tolerance during the constraint solving. (optional, default=0.0001)") )
    , d_activate( initData(&d_activate, true, "activate", "control constraint activation (true by default)"))
    , keepOrientDiff(initData(&keepOrientDiff,false, "keepOrientationDifference", "keep the initial difference in orientation (only for rigids)"))
    , l_topology1(initLink("topology1", "link to the first topology container"))
    , l_topology2(initLink("topology2", "link to the second topology container"))
{
    this->f_listening.setValue(true);
}

template<class DataTypes>
BilateralLagrangianConstraint<DataTypes>::BilateralLagrangianConstraint(MechanicalState* object)
    : BilateralLagrangianConstraint(object, object)
{
}

template<class DataTypes>
BilateralLagrangianConstraint<DataTypes>::BilateralLagrangianConstraint()
    : BilateralLagrangianConstraint(nullptr, nullptr)
{
}

template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::unspecializedInit()
{
    /// Do general check of validity for inputs
    Inherit1::init();

    /// Using assert means that the previous lines have check that there is two valid mechanical state.
    assert(this->mstate1);
    assert(this->mstate2);

    prevForces.clear();
}

template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::init()
{
    unspecializedInit();

    if (sofa::core::topology::BaseMeshTopology* _topology1 = l_topology1.get())
    {
        m1.createTopologyHandler(_topology1);
        m1.addTopologyEventCallBack(core::topology::TopologyChangeType::POINTSREMOVED,
            [this](const core::topology::TopologyChange* change)
        {
            const auto* pointsRemoved = static_cast<const core::topology::PointsRemoved*>(change);
            removeContact(0, pointsRemoved->getArray());
        });
    }

    if (sofa::core::topology::BaseMeshTopology* _topology2 = l_topology2.get())
    {
        m2.createTopologyHandler(_topology2);
        m2.addTopologyEventCallBack(core::topology::TopologyChangeType::POINTSREMOVED,
            [this](const core::topology::TopologyChange* change)
        {
            const auto* pointsRemoved = static_cast<const core::topology::PointsRemoved*>(change);
            removeContact(1, pointsRemoved->getArray());
        });
    }
}

template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::reinit()
{
    prevForces.clear();
}


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::buildConstraintMatrix(const ConstraintParams*, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &constraintId
                                                                      , const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/)
{
    if (!d_activate.getValue())
        return;

    const unsigned minp = std::min(m1.getValue().size(), m2.getValue().size());
    if (minp == 0)
        return;

    const SubsetIndices& m1Indices = m1.getValue();
    const SubsetIndices& m2Indices = m2.getValue();

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    cid.resize(minp);

    for (unsigned pid=0; pid<minp; pid++)
    {
        int tm1 = m1Indices[pid];
        int tm2 = m2Indices[pid];

        constexpr type::Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);

        cid[pid] = constraintId;
        constraintId += 3;

        MatrixDerivRowIterator c1_it = c1.writeLine(cid[pid]);
        c1_it.addCol(tm1, -cx);

        MatrixDerivRowIterator c2_it = c2.writeLine(cid[pid]);
        c2_it.addCol(tm2, cx);

        c1_it = c1.writeLine(cid[pid] + 1);
        c1_it.setCol(tm1, -cy);

        c2_it = c2.writeLine(cid[pid] + 1);
        c2_it.setCol(tm2, cy);

        c1_it = c1.writeLine(cid[pid] + 2);
        c1_it.setCol(tm1, -cz);

        c2_it = c2.writeLine(cid[pid] + 2);
        c2_it.setCol(tm2, cz);
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams,
                                                                       BaseVector *v,
                                                                       const DataVecCoord &d_x1, const DataVecCoord &d_x2
                                                                       , const DataVecDeriv & d_v1, const DataVecDeriv & d_v2)
{
    if (!d_activate.getValue()) return;

    const SubsetIndices& m1Indices = m1.getValue();
    const SubsetIndices& m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());

    const VecDeriv& restVector = this->restVector.getValue();

    if (cParams->constOrder() == sofa::core::ConstraintOrder::VEL)
    {
        getVelocityViolation(v, d_x1, d_x2, d_v1, d_v2);
        return;
    }

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

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


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::getVelocityViolation(BaseVector *v,
                                                                     const DataVecCoord &d_x1,
                                                                     const DataVecCoord &d_x2,
                                                                     const DataVecDeriv &d_v1,
                                                                     const DataVecDeriv &d_v2)
{
    const SubsetIndices& m1Indices = m1.getValue();
    const SubsetIndices& m2Indices = m2.getValue();

    SOFA_UNUSED(d_x1);
    SOFA_UNUSED(d_x2);

    const VecCoord &v1 = d_v1.getValue();
    const VecCoord &v2 = d_v2.getValue();

    const unsigned minp = std::min(m1Indices.size(), m2Indices.size());
    const VecDeriv& restVector = this->restVector.getValue();

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


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::getConstraintResolution(const ConstraintParams* cParams,
                                                                        std::vector<ConstraintResolution*>& resTab,
                                                                        unsigned int& offset)
{
    SOFA_UNUSED(cParams);
    const unsigned minp=std::min(m1.getValue().size(),m2.getValue().size());

    prevForces.resize(minp);
    for (unsigned pid=0; pid<minp; pid++)
    {
        resTab[offset] = new BilateralConstraintResolution3Dof(&prevForces[pid]);
        offset += 3;
    }
}

template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q,
                                                           Real /*contactDistance*/, int m1, int m2,
                                                           Coord /*Pfree*/, Coord /*Qfree*/,
                                                           long /*id*/, PersistentID /*localid*/)
{
    WriteAccessor<Data<SubsetIndices> > wm1 = this->m1;
    WriteAccessor<Data<SubsetIndices> > wm2 = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);
    wrest.push_back(Q-P);
}


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::addContact(Deriv norm, Coord P, Coord Q, Real
                                                           contactDistance, int m1, int m2,
                                                           long id, PersistentID localid)
{
   addContact(norm, P, Q, contactDistance, m1, m2,
               this->getMState2()->read(ConstVecCoordId::freePosition())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::freePosition())->getValue()[m1],
               id, localid);
}

template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::addContact(Deriv norm, Real contactDistance,
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


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::removeContact(int objectId, SubsetIndices indices)
{
    WriteAccessor<Data <SubsetIndices > > m1Indices = this->m1;
    WriteAccessor<Data <SubsetIndices > > m2Indices = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;

    const SubsetIndices& cIndices1 = m1.getValue();
    const SubsetIndices& cIndices2 = m2.getValue();

    for (sofa::Size i = 0; i < indices.size(); ++i)
    {
        const Index elemId = indices[i];
        Index posId = sofa::InvalidID;
            
        if (objectId == 0)
            posId = indexOfElemConstraint(cIndices1, elemId);
        else if (objectId == 1)
            posId = indexOfElemConstraint(cIndices2, elemId);

        if (posId != sofa::InvalidID)
        {
            if (wrest.size() == m1Indices.size())
                wrest.erase(wrest.begin() + posId);
            
            m1Indices.erase(m1Indices.begin() + posId);
            m2Indices.erase(m2Indices.begin() + posId);
        }
    }
    
}


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::clear(int reserve)
{
    WriteAccessor<Data <SubsetIndices > > wm1 = this->m1;
    WriteAccessor<Data <SubsetIndices > > wm2 = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.clear();
    wm2.clear();
    wrest.clear();
    if (reserve)
    {
        wm1.reserve(reserve);
        wm2.reserve(reserve);
        wrest.reserve(reserve);
    }
}


template<class DataTypes>
Index BilateralLagrangianConstraint<DataTypes>::indexOfElemConstraint(const SubsetIndices& cIndices, Index Id)
{
    const auto it = std::find(cIndices.begin(), cIndices.end(), Id);

    if (it != cIndices.end())
        return Index(std::distance(cIndices.begin(), it));
    else    
        return sofa::InvalidID;
}


template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    constexpr sofa::type::RGBAColor colorActive = sofa::type::RGBAColor::magenta();
    constexpr sofa::type::RGBAColor colorNotActive = sofa::type::RGBAColor::green();
    std::vector< sofa::type::Vec3 > vertices;

    const unsigned minp = std::min(m1.getValue().size(),m2.getValue().size());
    auto positionsM1 = sofa::helper::getReadAccessor(*this->mstate1->read(ConstVecCoordId::position()));
    auto positionsM2 = sofa::helper::getReadAccessor(*this->mstate2->read(ConstVecCoordId::position()));
    const auto indicesM1 = sofa::helper::getReadAccessor(m1);
    const auto indicesM2 = sofa::helper::getReadAccessor(m2);

    for (unsigned i=0; i<minp; i++)
    {
        vertices.push_back(DataTypes::getCPos(positionsM1[indicesM1[i]]));
        vertices.push_back(DataTypes::getCPos(positionsM2[indicesM2[i]]));
    }

    vparams->drawTool()->drawPoints(vertices, 10, (d_activate.getValue()) ? colorActive : colorNotActive);


}

//TODO(dmarchal): implementing keyboard interaction behavior directly in a component is not a valid
//design for a component. Interaction should be defered to an independent Component implemented in the SofaInteraction
//a second possibility is to implement this behavir using script.
template<class DataTypes>
void BilateralLagrangianConstraint<DataTypes>::handleEvent(Event *event)
{
    if (KeypressedEvent::checkEventType(event))
    {
        const KeypressedEvent *ev = static_cast<KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'A':
        case 'a':
            if (d_activate.getValue())
            {
                msg_info() << "Unactivating constraint";
                d_activate.setValue(false);
            }
            else
            {
                msg_info() << "Activating constraint";
                d_activate.setValue(true);
            }
            
            break;
        }
    }

}


} //namespace sofa::component::constraint::lagrangian::model
