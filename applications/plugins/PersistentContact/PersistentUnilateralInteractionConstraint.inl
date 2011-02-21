/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL

#include "PersistentUnilateralInteractionConstraint.h"

#include <sofa/component/constraintset/UnilateralInteractionConstraint.inl>

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
void PersistentUnilateralInteractionConstraint<DataTypes>::addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id, PersistentID localid, bool isPersistent)
{
    // Compute dt and delta
    const Real delta = dot(P-Q, norm) - contactDistance;
    const Real deltaFree = dot(Pfree-Qfree, norm) - contactDistance;

    if (this->f_printLog.getValue())
    {
        std::cout << "delta : " << delta << " - deltaFree : " << deltaFree << std::endl;
        std::cout << "P : " << P << " - PFree : " << Pfree << std::endl;
        std::cout << "Q : " << Q << " - QFree : " << Qfree << std::endl;
    }

    unsigned int lastContactIndex = this->contacts.size();
    this->contacts.resize(lastContactIndex + 1);

    typename Inherited::Contact& c = this->contacts[lastContactIndex];

    c.P				= P;
    c.Q				= Q;
    c.Pfree			= Pfree;
    c.Qfree			= Qfree;
    c.m1			= m1;
    c.m2			= m2;
    c.norm			= norm;
    c.delta			= delta;
    c.mu			= mu;
    c.contactId		= id;
    c.localId		= localid;
    c.t				= Deriv(norm.z(), norm.x(), norm.y());
    c.s				= cross(norm, c.t);
    c.s				= c.s / c.s.norm();
    c.t				= cross((-norm), c.s);

    const Deriv PPfree = Pfree - P;
    const Deriv QQfree = Qfree - Q;
    const Real REF_DIST = PPfree.norm() + QQfree.norm();

    if (isPersistent)
    {

        std::cout << "Persistent contact: \nP : " << P << " - PFree : " << Pfree << std::endl;
        std::cout << "Q : " << Q << " - QFree : " << Qfree << std::endl;
        c.dfree		= deltaFree;
        c.dfree_t	= dot(Pfree - Qfree, c.t);
        c.dfree_s	= dot(Pfree - Qfree, c.s);

        return;
    }

    if (helper::rabs(delta) < 0.00001 * REF_DIST)
    {
        c.dfree		= deltaFree;
        c.dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
        c.dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);

        return;
    }

    if (helper::rabs(delta - deltaFree) > 0.001 * delta)
    {
        const Real dt = delta / (delta - deltaFree);

        if (dt > 0.0 && dt < 1.0)
        {
            const sofa::defaulttype::Vector3 Qt = Q * (1-dt) + Qfree * dt;
            const sofa::defaulttype::Vector3 Pt = P * (1-dt) + Pfree * dt;

            c.dfree		= deltaFree;
            c.dfree_t	= dot(Pfree-Pt, c.t) - dot(Qfree-Qt, c.t);
            c.dfree_s	= dot(Pfree-Pt, c.s) - dot(Qfree-Qt, c.s);
        }
        else
        {
            if (deltaFree < 0.0)
            {
                c.dfree		= deltaFree;
                c.dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
                c.dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);
            }
            else
            {
                c.dfree		= deltaFree;
                c.dfree_t	= 0;
                c.dfree_s	= 0;
            }
        }
    }
    else
    {
        c.dfree		= deltaFree;
        c.dfree_t	= dot(PPfree, c.t) - dot(QQfree, c.t);
        c.dfree_s	= dot(PPfree, c.s) - dot(QQfree, c.s);
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
bool PersistentUnilateralInteractionConstraint<DataTypes>::isSticked(int _contactId)
{
    typename sofa::helper::vector< Contact >::iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
        return (contactStates[it->id] == PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY);
    else
        return false;
}


template<class DataTypes>
bool PersistentUnilateralInteractionConstraint<DataTypes>::isSliding(int _contactId)
{
    typename sofa::helper::vector< Contact >::iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
        return (contactStates[it->id] == PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING);
    else
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
typename PersistentUnilateralInteractionConstraint<DataTypes>::Deriv PersistentUnilateralInteractionConstraint<DataTypes>::getContactForce(int _contactId)
{
    typename sofa::helper::vector< Contact >::iterator it = this->contacts.begin();
    typename sofa::helper::vector< Contact >::iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
    {
        if (contactForces.find(it->id) != contactForces.end())
        {
            return contactForces[it->id];
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
void PersistentUnilateralInteractionConstraint<DataTypes>::debugContactStates()
{
    std::cout << "-------------->debugContactStates\n";

    typename std::map< int, ContactState >::iterator it = contactStates.begin();
    typename std::map< int, ContactState >::iterator itEnd = contactStates.end();

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
void PersistentUnilateralInteractionConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);

    for (unsigned int i=0; i< this->contacts.size(); i++)
    {
        const Contact& c = this->contacts[i];

        glLineWidth(5);
        glBegin(GL_LINES);

        switch (contactStates[c.id])
        {
        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::NONE :
            glColor4f(1,0,0,1);
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING :
            glColor4f(0,0,1,1);
            break;

        case PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY :
            glColor4f(0,1,0,1);
            break;
        }

        helper::gl::glVertexT(c.P);
        helper::gl::glVertexT(c.Q);

        glEnd();

        glLineWidth(1);
        glBegin(GL_LINES);

        glColor4f(1,1,1,1);
        helper::gl::glVertexT(c.Pfree);
        helper::gl::glVertexT(c.Qfree);

        glEnd();

        glLineWidth(1);
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_INL
