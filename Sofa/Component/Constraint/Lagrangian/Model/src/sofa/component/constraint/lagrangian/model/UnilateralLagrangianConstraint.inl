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
#include <sofa/component/constraint/lagrangian/model/UnilateralLagrangianConstraint.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
UnilateralLagrangianConstraint<DataTypes>::UnilateralLagrangianConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , epsilon(Real(0.001))
    , yetIntegrated(false)
    , customTolerance(0.0)
    , contactsStatus(nullptr)
{
}

template<class DataTypes>
UnilateralLagrangianConstraint<DataTypes>::~UnilateralLagrangianConstraint()
{
    if(contactsStatus)
        delete[] contactsStatus;
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::clear(int reserve)
{
    contacts.clear();
    if (reserve)
        contacts.reserve(reserve);
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::addContact(SReal mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, long id, PersistentID localid)
{
    addContact(mu, norm, P, Q, contactDistance, m1, m2,
            this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue()[m2],
            this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue()[m1],
            id, localid);
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::addContact(SReal mu, Deriv norm, Real contactDistance, int m1, int m2, long id, PersistentID localid)
{
    addContact(mu, norm,
            this->getMState2()->read(core::ConstVecCoordId::position())->getValue()[m2],
            this->getMState1()->read(core::ConstVecCoordId::position())->getValue()[m1],
            contactDistance, m1, m2,
            this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue()[m2],
            this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue()[m1],
            id, localid);
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::addContact(SReal mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord /*Pfree*/, Coord /*Qfree*/, long id, PersistentID localid)
{
    contacts.resize(contacts.size() + 1);
    Contact &c = contacts.back();

    c.P			= P;
    c.Q			= Q;
    c.m1		= m1;
    c.m2		= m2;
    c.norm		= norm;
    c.t			= Deriv(norm.z(), norm.x(), norm.y());
    c.s			= cross(norm, c.t);
    c.s			= c.s / c.s.norm();
    c.t			= cross((-norm), c.s);
    c.mu		= mu;
    c.contactId = id;
    c.localId	= localid;
    c.contactDistance = contactDistance;
}


template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams *, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &contactId
        , const DataVecCoord &, const DataVecCoord &)
{
    assert(this->mstate1);
    assert(this->mstate2);

    if (this->mstate1 == this->mstate2)
    {
        MatrixDeriv& c1 = *c1_d.beginEdit();

        for (unsigned int i = 0; i < contacts.size(); i++)
        {
            Contact& c = contacts[i];

            c.id = contactId++;

            MatrixDerivRowIterator c1_it = c1.writeLine(c.id);

            c1_it.addCol(c.m1, -c.norm);
            c1_it.addCol(c.m2, c.norm);

            if (c.mu > 0.0)
            {
                c1_it = c1.writeLine(c.id + 1);
                c1_it.setCol(c.m1, -c.t);
                c1_it.setCol(c.m2, c.t);

                c1_it = c1.writeLine(c.id + 2);
                c1_it.setCol(c.m1, -c.s);
                c1_it.setCol(c.m2, c.s);

                contactId += 2;
            }
        }

        c1_d.endEdit();
    }
    else
    {
        MatrixDeriv& c1 = *c1_d.beginEdit();
        MatrixDeriv& c2 = *c2_d.beginEdit();

        for (unsigned int i = 0; i < contacts.size(); i++)
        {
            Contact& c = contacts[i];

            c.id = contactId++;

            MatrixDerivRowIterator c1_it = c1.writeLine(c.id);
            c1_it.addCol(c.m1, -c.norm);

            MatrixDerivRowIterator c2_it = c2.writeLine(c.id);
            c2_it.addCol(c.m2, c.norm);

            if (c.mu > 0.0)
            {
                c1_it = c1.writeLine(c.id + 1);
                c1_it.setCol(c.m1, -c.t);

                c1_it = c1.writeLine(c.id + 2);
                c1_it.setCol(c.m1, -c.s);

                c2_it = c2.writeLine(c.id + 1);
                c2_it.setCol(c.m2, c.t);

                c2_it = c2.writeLine(c.id + 2);
                c2_it.setCol(c.m2, c.s);

                contactId += 2;
            }
        }

        c1_d.endEdit();
        c2_d.endEdit();
    }
}


template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::getPositionViolation(linearalgebra::BaseVector *v)
{
    const VecCoord &PfreeVec = this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue();
    const VecCoord &QfreeVec = this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue();

    Real dfree = (Real)0.0;
    Real dfree_t = (Real)0.0;
    Real dfree_s = (Real)0.0;

    const unsigned int cSize = contacts.size();

    for (unsigned int i = 0; i < cSize; i++)
    {
        const Contact& c = contacts[i];

        // Compute dfree, dfree_t and d_free_s

        const Coord &Pfree =  PfreeVec[c.m2];
        const Coord &Qfree =  QfreeVec[c.m1];

        const Coord PPfree = Pfree - c.P;
        const Coord QQfree = Qfree - c.Q;

        const Real ref_dist = PPfree.norm() + QQfree.norm();

        dfree = dot(Pfree - Qfree, c.norm) - c.contactDistance;
        const Real delta = dot(c.P - c.Q, c.norm) - c.contactDistance;

        if ((helper::rabs(delta) < 0.00001 * ref_dist) && (helper::rabs(dfree) < 0.00001 * ref_dist))
        {
            dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
        }
        else if (helper::rabs(delta - dfree) > 0.001 * delta)
        {
            const Real dt = delta / (delta - dfree);

            if (dt > 0.0 && dt < 1.0)
            {
                const Coord Pt		= c.P * (1 - dt) + Pfree * dt;
                const Coord Qt		= c.Q * (1 - dt) + Qfree * dt;
                const Coord PtPfree = Pfree - Pt;
                const Coord QtQfree = Qfree - Qt;

                dfree_t = dot(PtPfree, c.t) - dot(QtQfree, c.t);
                dfree_s = dot(PtPfree, c.s) - dot(QtQfree, c.s);
            }
            else if (dfree < 0.0)
            {
                dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
                dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
            }
            else
            {
                dfree_t = 0;
                dfree_s = 0;
            }
        }
        else
        {
            dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
        }

        // Sets dfree in global violation vector

        v->set(c.id, dfree);

        c.dfree = dfree; // PJ : For isActive() method. Don't know if it's still usefull.

        if (c.mu > 0.0)
        {
            v->set(c.id + 1, dfree_t);
            v->set(c.id + 2, dfree_s);
        }
    }
}


template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::getVelocityViolation(linearalgebra::BaseVector *v)
{
    auto P = this->getMState2()->readPositions();
    auto Q = this->getMState1()->readPositions();

    const SReal dt = this->getContext()->getDt();
    const SReal invDt = SReal(1.0) / dt;

    const VecDeriv &PvfreeVec = this->getMState2()->read(core::ConstVecDerivId::freeVelocity())->getValue();
    const VecDeriv &QvfreeVec = this->getMState1()->read(core::ConstVecDerivId::freeVelocity())->getValue();

    const unsigned int cSize = contacts.size();

    for (unsigned int i = 0; i < cSize; i++)
    {
        const Contact& c = contacts[i];

        const Deriv QP_invDt = (P[c.m2] - Q[c.m1])*invDt;
        const Deriv QP_vfree  = PvfreeVec[c.m2] - QvfreeVec[c.m1];
        const Deriv dFreeVec = QP_vfree + QP_invDt;

        v->set(c.id, dot(dFreeVec, c.norm) - c.contactDistance*invDt ); // dvfree = 1/dt *  [ dot ( P - Q, n) - contactDist ] + dot(v_P - v_Q , n ) ]  

        if (c.mu > 0.0)
        {
            v->set(c.id + 1, dot(QP_vfree, c.t)); // dfree_t
            v->set(c.id + 2, dot(QP_vfree, c.s)); // dfree_s
        }
    }
}


template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams *cparams, linearalgebra::BaseVector *v, const DataVecCoord &, const DataVecCoord &
        , const DataVecDeriv &, const DataVecDeriv &)
{
    switch (cparams->constOrder())
    {
    case core::ConstraintOrder::POS_AND_VEL :
    case core::ConstraintOrder::POS :
        getPositionViolation(v);
        break;

    case core::ConstraintOrder::ACC :
    case core::ConstraintOrder::VEL :
        getVelocityViolation(v);
        break;

    default :
        msg_error() << "UnilateralLagrangianConstraint doesn't implement " << cparams->getName() << " constraint violation\n";
        break;
    }
}


template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::getConstraintInfo(const core::ConstraintParams*, VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& /*positions*/, VecConstDeriv& directions, VecConstArea& /*areas*/)
{
    if (contacts.empty()) return;
    const bool friction = (contacts[0].mu > 0.0); /// @todo: can there be both friction-less and friction contacts in the same UnilateralLagrangianConstraint ???
    ConstraintBlockInfo info;
    info.parent = this;
    info.const0 = contacts[0].id;
    info.nbLines = friction ? 3 : 1;
    info.hasId = true;
    info.offsetId = ids.size();
    info.hasDirection = true;
    info.offsetDirection = directions.size();
    info.nbGroups = contacts.size();

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        ids.push_back( yetIntegrated ? c.contactId : -c.contactId);
        directions.push_back( c.norm );
        if (friction)
        {
            directions.push_back( c.t );
            directions.push_back( c.s );
        }
    }

    yetIntegrated = true;

    blocks.push_back(info);
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if(contactsStatus)
    {
        delete[] contactsStatus;
        contactsStatus = nullptr;
    }

    if (contacts.size() > 0)
    {
        contactsStatus = new bool[contacts.size()];
        memset(contactsStatus, 0, sizeof(bool)*contacts.size());
    }

    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if(c.mu > 0.0)
        {
            UnilateralConstraintResolutionWithFriction* ucrwf = new UnilateralConstraintResolutionWithFriction(c.mu, nullptr, &contactsStatus[i]);
            ucrwf->setTolerance(customTolerance);
            resTab[offset] = ucrwf;

            // TODO : cette méthode de stockage des forces peu mal fonctionner avec 2 threads quand on utilise l'haptique
            offset += 3;
        }
        else
            resTab[offset++] = new UnilateralConstraintResolution();
    }
}

template<class DataTypes>
bool UnilateralLagrangianConstraint<DataTypes>::isActive() const
{
    for(unsigned int i = 0; i < contacts.size(); i++)
        if(contacts[i].dfree < 0)
            return true;

    return false;
}

template<class DataTypes>
void UnilateralLagrangianConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> redVertices;
    std::vector<sofa::type::Vec3> otherVertices;
    std::vector<sofa::type::RGBAColor> otherColors;

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];

        otherVertices.push_back(c.P);        
        otherVertices.push_back(c.P + c.norm);
        otherColors.push_back(sofa::type::RGBAColor::white());

        otherVertices.push_back(c.Q);
        otherVertices.push_back(c.Q - c.norm);
        otherColors.push_back(sofa::type::RGBAColor(0,0.5,0.5,1));

    }
    vparams->drawTool()->drawLines(otherVertices, 3, otherColors);




}

} //namespace sofa::component::constraint::lagrangian::model
