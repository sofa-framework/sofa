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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL

#include "PersistentUnilateralInteractionConstraint.h"

#include <SofaConstraint/UnilateralInteractionConstraint.inl>

#include <sofa/helper/rmath.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

template< class DataTypes >
void PersistentUnilateralConstraintResolutionWithFriction< DataTypes >::init(int line, double** w, double* force)
{
    _W[0] = w[line		][line		];
    _W[1] = w[line		][line + 1	];
    _W[2] = w[line		][line + 2	];
    _W[3] = w[line + 1	][line + 1	];
    _W[4] = w[line + 1	][line + 2	];
    _W[5] = w[line + 2	][line + 2	];

    ///@TODO OPTIMIZATION
    force[line		] = _f[0];
    force[line + 1	] = _f[1];
    force[line + 2	] = _f[2];
}

template< class DataTypes >
void PersistentUnilateralConstraintResolutionWithFriction< DataTypes >::resolution(int line, double** /*w*/, double* d, double* force, double * /*dfree*/)
{
    double f[2];
    double normFt;

    f[0] = force[line]; f[1] = force[line+1];
    force[line] -= d[line] / _W[0];

    if(force[line] < 0)
    {
        force[line]=0; force[line+1]=0; force[line+2]=0;

        ///@TODO OPTIMIZATION
        /*if (m_constraint)
        	m_constraint->setContactState(line, NONE);*/

        return;
    }

    d[line+1] += _W[1] * (force[line]-f[0]);
    d[line+2] += _W[2] * (force[line]-f[0]);
    force[line+1] -= 2*d[line+1] / (_W[3] +_W[5]) ;
    force[line+2] -= 2*d[line+2] / (_W[3] +_W[5]) ;

    normFt = sqrt(force[line+1]*force[line+1] + force[line+2]*force[line+2]);

    if(normFt > _mu*force[line])
    {
        force[line+1] *= _mu*force[line]/normFt;
        force[line+2] *= _mu*force[line]/normFt;

        ///@TODO OPTIMIZATION
        /*if (m_constraint)
        {
        	m_constraint->setContactState(line, SLIDING);
        	m_constraint->setContactForce(line, Deriv(force[line], force[line+1], force[line+2]));
        }*/
    }
    else
    {
        ///@TODO OPTIMIZATION
        /*if (m_constraint)
        {
        	m_constraint->setContactState(line, STICKY);
        	m_constraint->setContactForce(line, Deriv(force[line], force[line+1], force[line+2]));
        }*/
    }
}

template< class DataTypes >
void PersistentUnilateralConstraintResolutionWithFriction< DataTypes >::store(int line, double* force, bool /*convergence*/)
{
    if (m_constraint)
    {
        if(force[line] < 0)
        {
            m_constraint->setContactState(line, NONE);
        }
        else
        {
            double normFt = sqrt(force[line+1]*force[line+1] + force[line+2]*force[line+2]);

            if(normFt > _mu*force[line])
            {
                m_constraint->setContactState(line, SLIDING);
                m_constraint->setContactForce(line, Deriv(force[line], force[line+1], force[line+2]));
            }
            else
            {
                m_constraint->setContactState(line, STICKY);
                m_constraint->setContactForce(line, Deriv(force[line], force[line+1], force[line+2]));
            }
        }
    }

    m_constraint = 0;

    if(_active)
    {
        *_active = (force[line] != 0);

        // Won't be used in the haptic thread
        _active = NULL;
    }
}


template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::addContact(double mu, Deriv norm, Real cDist, int m1, int m2, long id, PersistentID localid)
{
    unsigned int lastContactIndex = this->contacts.size();
    this->contacts.resize(lastContactIndex + 1);

    typename Inherited::Contact& c = this->contacts[lastContactIndex];

    c.P					= this->getMState2()->read(core::ConstVecCoordId::position())->getValue()[m2];
    c.Q					= this->getMState1()->read(core::ConstVecCoordId::position())->getValue()[m1];
    c.m1				= m1;
    c.m2				= m2;
    c.norm				= norm;
    c.t					= Deriv(norm.z(), norm.x(), norm.y());
    c.s					= cross(norm, c.t);
    c.s					= c.s / c.s.norm();
    c.t					= cross((-norm), c.s);
    c.mu				= mu;
    c.contactId			= id;
    c.localId			= localid;
    c.contactDistance	= cDist;
}


template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::getPositionViolation(defaulttype::BaseVector *v)
{
    const VecCoord &PfreeVec = this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue();
    const VecCoord &QfreeVec = this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue();

    Real dfree		= (Real)0.0;
    Real dfree_t	= (Real)0.0;
    Real dfree_s	= (Real)0.0;

    const unsigned int cSize = this->contacts.size();

    for (unsigned int i = 0; i < cSize; i++)
    {
        const Contact& c = this->contacts[i];

        // Compute dfree, dfree_t and d_free_s

        const Coord &Pfree =  PfreeVec[c.m2];
        const Coord &Qfree =  QfreeVec[c.m1];

        const Coord PPfree = Pfree - c.P;
        const Coord QQfree = Qfree - c.Q;

        const Real REF_DIST = PPfree.norm() + QQfree.norm();

        dfree = dot(Pfree - Qfree, c.norm) - c.contactDistance;
        const Real delta = dot(c.P - c.Q, c.norm) - c.contactDistance;

        if (helper::rabs(delta) < 0.00001 * REF_DIST)
        {
            dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);
        }
        else if (helper::rabs(delta - dfree) > 0.001 * delta)
        {
            const Real dt = delta / (delta - dfree);

            if (dt > 0.0 && dt < 1.0)
            {
                const sofa::defaulttype::Vector3 Qt = c.Q * (1-dt) + Qfree * dt;
                const sofa::defaulttype::Vector3 Pt = c.P * (1-dt) + Pfree * dt;

                dfree_t	= dot(Pfree-Pt, c.t) - dot(Qfree-Qt, c.t);
                dfree_s	= dot(Pfree-Pt, c.s) - dot(Qfree-Qt, c.s);
            }
            else
            {
                if (dfree < 0.0)
                {
                    dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
                    dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);
                }
                else
                {
                    dfree_t	= 0;
                    dfree_s	= 0;
                }
            }
        }
        else
        {
            dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);
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
void PersistentUnilateralInteractionConstraint<DataTypes>::getVelocityViolation(defaulttype::BaseVector *v)
{
    const VecDeriv &PvfreeVec = this->getMState2()->read(core::ConstVecDerivId::freeVelocity())->getValue();
    const VecDeriv &QvfreeVec = this->getMState1()->read(core::ConstVecDerivId::freeVelocity())->getValue();

    const unsigned int cSize = this->contacts.size();

    for (unsigned int i = 0; i < cSize; i++)
    {
        const Contact& c = this->contacts[i];

        const Deriv QP_vfree = PvfreeVec[c.m2] - QvfreeVec[c.m1];

        v->set(c.id, dot(QP_vfree, c.norm)); // dfree

        if (c.mu > 0.0)
        {
            v->set(c.id + 1, dot(QP_vfree, c.t)); // dfree_t
            v->set(c.id + 2, dot(QP_vfree, c.s)); // dfree_s
        }
    }
}


template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if (this->contactsStatus)
        delete[] this->contactsStatus;

    this->contactsStatus = new bool[this->contacts.size()];
    memset(this->contactsStatus, 0, sizeof(bool)*this->contacts.size());

    for(unsigned int i=0; i<this->contacts.size(); i++)
    {
        Contact& c = this->contacts[i];

        if (c.mu > 0.0)
        {
            PersistentUnilateralConstraintResolutionWithFriction<DataTypes> *cRes = new PersistentUnilateralConstraintResolutionWithFriction<DataTypes>(c.mu, &this->contactsStatus[i]);
            cRes->setConstraint(this);
            cRes->setInitForce(getInitForce(c.contactId));
            resTab[offset] = cRes;
            offset += 3;
        }
        else
            resTab[offset++] = new UnilateralConstraintResolution();
    }
}


template<class DataTypes>
bool PersistentUnilateralInteractionConstraint<DataTypes>::isSticked(int _contactId) const
{
    typename sofa::helper::vector< Contact >::const_iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::const_iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
    {
        contactStateIterator contactStateIt = contactStates.find(it->id);

        if (contactStateIt != contactStates.end())
            return (contactStateIt->second == PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY);
    }

    return false;
}


template<class DataTypes>
bool PersistentUnilateralInteractionConstraint<DataTypes>::isSliding(int _contactId) const
{
    typename sofa::helper::vector< Contact >::const_iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::const_iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
    {
        contactStateIterator contactStateIt = contactStates.find(it->id);

        if (contactStateIt != contactStates.end())
            return (contactStateIt->second == PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING);
    }

    return false;
}


template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::setContactState(int id, ContactState s)
{
    if (contactStates.find(id) != contactStates.end())
    {
        contactStates[id] = s;
    }
    else
    {
        contactStates.insert(std::make_pair(id, s));
    }
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::clearContactStates()
{
    contactStates.clear();
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::setContactForce(int id, Deriv f)
{
    if (contactForces.find(id) != contactForces.end())
    {
        contactForces[id] = f;
    }
    else
    {
        contactForces.insert(std::make_pair(id, f));
    }
}

template<class DataTypes>
typename PersistentUnilateralInteractionConstraint<DataTypes>::Deriv PersistentUnilateralInteractionConstraint<DataTypes>::getContactForce(int _contactId) const
{
    typename sofa::helper::vector< Contact >::const_iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::const_iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
    {
        typename std::map< int, Deriv >::const_iterator contactForcesIt = contactForces.find(it->id);

        if (contactForcesIt != contactForces.end())
        {
            return contactForcesIt->second;
        }
    }

    return Deriv();
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::clearContactForces()
{
    contactForces.clear();
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::setInitForce(int _contactId, Deriv f)
{
    if (initForces.find(_contactId) != initForces.end())
    {
        initForces[_contactId] = f;
    }
    else
    {
        initForces.insert(std::make_pair(_contactId, f));
    }
}

template<class DataTypes>
typename PersistentUnilateralInteractionConstraint<DataTypes>::Deriv PersistentUnilateralInteractionConstraint<DataTypes>::getInitForce(int _contactId)
{
    if (initForces.find(_contactId) != initForces.end())
    {
        return initForces[_contactId];
    }

    return Deriv();
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::clearInitForces()
{
    initForces.clear();
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::debugContactStates() const
{
    std::cout << "-------------->debugContactStates\n";

    typename std::map< int, ContactState >::const_iterator it = contactStates.begin();
    typename std::map< int, ContactState >::const_iterator itEnd = contactStates.end();

    std::string s;

    while (it != itEnd)
    {
        switch (it->second)
        {
        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::NONE :
            s = "NONE";
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING :
            s = "SLIDING";
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY :
            s = "STICKY";
            break;
        }

        std::cout << it->first << " : " << s << std::endl;
        ++it;
    }
}

template<class DataTypes>
void PersistentUnilateralInteractionConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    return;
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);

    for (unsigned int i=0; i< this->contacts.size(); i++)
    {
        const Contact& c = this->contacts[i];

        glLineWidth(5);
        glBegin(GL_LINES);

        switch (contactStates[c.id])
        {
        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::NONE :
            glColor4f(1.f,1.f,1.f,1.f);
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING :
            glColor4f(0.05f,0.05f,0.8f,1.f);
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY :
            glColor4f(0.8f,0.05f,0.05f,1.f);
            break;
        }

        helper::gl::glVertexT(c.P);
        helper::gl::glVertexT(c.Q);

        glEnd();

        /*glLineWidth(2);
        glBegin(GL_LINES);

        glColor4f(0.8f,0.8f,0.8f,1.f);
        helper::gl::glVertexT(c.Pfree);
        helper::gl::glVertexT(c.Qfree);

        glEnd();*/

        glLineWidth(1);
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL
