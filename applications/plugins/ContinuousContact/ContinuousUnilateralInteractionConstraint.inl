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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_INL

#include "ContinuousUnilateralInteractionConstraint.h"

#include <sofa/component/constraintset/UnilateralInteractionConstraint.inl>

#include <sofa/helper/rmath.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

template< class DataTypes >
void ContinuousUnilateralConstraintResolutionWithFriction< DataTypes >::init(int line, double** w, double* force)
{
    _W[0]=w[line  ][line  ];
    _W[1]=w[line  ][line+1];
    _W[2]=w[line  ][line+2];
    _W[3]=w[line+1][line+1];
    _W[4]=w[line+1][line+2];
    _W[5]=w[line+2][line+2];

//	return;

    ////////////////// christian : the following does not work ! /////////
    if(_prev)
    {
        force[line] = _prev->popForce();
        force[line+1] = _prev->popForce();
        force[line+2] = _prev->popForce();
    }
}

template< class DataTypes >
void ContinuousUnilateralConstraintResolutionWithFriction< DataTypes >::resolution(int line, double** /*w*/, double* d, double* force, double * /*dfree*/)
{
    double f[2];
    double normFt;

    f[0] = force[line]; f[1] = force[line+1];
    force[line] -= d[line] / _W[0];

    if(force[line] < 0)
    {
        force[line]=0; force[line+1]=0; force[line+2]=0;

        if (m_constraint)
            m_constraint->setContactState(line, NONE);

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

        if (m_constraint)
            m_constraint->setContactState(line, SLIDING);
    }
    else
    {
        if (m_constraint)
            m_constraint->setContactState(line, STICKY);
    }
}

template< class DataTypes >
void ContinuousUnilateralConstraintResolutionWithFriction< DataTypes >::store(int line, double* force, bool /*convergence*/)
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
        _active = NULL; // Won't be used in the haptic thread
    }
}


template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id, PersistentID localid)
{
    // compute dt and delta
    Real delta = dot(P-Q, norm) - contactDistance;
    Real deltaFree = dot(Pfree-Qfree, norm) - contactDistance;
    Real dt;
    int i = this->contacts.size();
    this->contacts.resize(i+1);
    typename Inherited::Contact& c = this->contacts[i];

    std::cout<<"delta : "<<delta<<" - deltaFree : "<<deltaFree <<std::endl;
    std::cout<<"P : "<<P<<" - PFree : "<<Pfree <<std::endl;
    std::cout<<"Q : "<<Q<<" - QFree : "<<Qfree <<std::endl;


// for visu
    c.P = P;
    c.Q = Q;
    c.Pfree = Pfree;
    c.Qfree = Qfree;
//
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.delta = delta;
    c.t = Deriv(norm.z(), norm.x(), norm.y());
    c.s = cross(norm,c.t);
    c.s = c.s / c.s.norm();
    c.t = cross((-norm), c.s);
    c.mu = mu;
    c.contactId = id;
    c.localId = localid;

    Deriv PPfree = Pfree-P;
    Deriv QQfree = Qfree-Q;
    Real ref_dist = PPfree.norm()+QQfree.norm();

    if (helper::rabs(delta) < 0.00001*ref_dist )
    {
        dt=0.0;
        c.dfree = deltaFree;
        c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
        c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);

        return;
    }

    if (helper::rabs(delta - deltaFree) > 0.001 * delta)
    {
        dt = delta / (delta - deltaFree);
        if (dt > 0.0 && dt < 1.0  )
        {
            sofa::defaulttype::Vector3 Qt, Pt;
            Qt = Q*(1-dt) + Qfree*dt;
            Pt = P*(1-dt) + Pfree*dt;
            c.dfree = deltaFree;// dot(Pfree-Pt, c.norm) - dot(Qfree-Qt, c.norm);
            c.dfree_t = dot(Pfree-Pt, c.t) - dot(Qfree-Qt, c.t);
            c.dfree_s = dot(Pfree-Pt, c.s) - dot(Qfree-Qt, c.s);
            //printf("\n ! dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
        }
        else
        {
            if (deltaFree < 0.0)
            {
                dt=0.0;
                c.dfree = deltaFree; // dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
                //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
                c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
                c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);
            }
            else
            {
                dt=1.0;
                c.dfree = deltaFree;
                c.dfree_t = 0;
                c.dfree_s = 0;
            }
        }
    }
    else
    {
        dt = 0;
        c.dfree = deltaFree;
        c.dfree_t = 0;
        c.dfree_s = 0;
        //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
    }
    //sout<<"R_nts = ["<<c.norm<<" ; "<<c.t<<" ; "<<c.s<<" ];"<<sendl;
}


template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if (this->contactsStatus)
        delete[] this->contactsStatus;

    this->contactsStatus = new bool[this->contacts.size()];
    memset(this->contactsStatus, 0, sizeof(bool)*this->contacts.size());

    for(unsigned int i=0; i<this->contacts.size(); i++)
    {
        Contact& c = this->contacts[i];
        if(c.mu > 0.0)
        {
            ContinuousUnilateralConstraintResolutionWithFriction<DataTypes> *cRes = new ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>(c.mu, NULL, &this->contactsStatus[i]);
            cRes->setConstraint(this);
            resTab[offset] = cRes;
            offset += 3;
        }
        else
            resTab[offset++] = new UnilateralConstraintResolution();
    }
}


template<class DataTypes>
bool ContinuousUnilateralInteractionConstraint<DataTypes>::isSticked(int _contactId)
{
    sofa::helper::vector< Contact >::iterator it = this->contacts.begin();
    sofa::helper::vector< Contact >::iterator itEnd = this->contacts.end();

    while (it != itEnd)
    {
        if (it->contactId == _contactId)
            break;

        ++it;
    }

    if (it != itEnd)
        return (contactStates[it->id] == ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY);
    else
        return false;
}

template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::setContactState(int id, ContactState s)
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
void ContinuousUnilateralInteractionConstraint<DataTypes>::clearContactStates()
{
    contactStates.clear();
}

template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::debugContactStates()
{
    std::cout << "-------------->debugContactStates\n";

    std::map< int, ContactState >::iterator it = contactStates.begin();
    std::map< int, ContactState >::iterator itEnd = contactStates.end();

    std::string s;

    while (it != itEnd)
    {
        switch (it->second)
        {
        case ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>::NONE :
            s = "NONE";
            break;

        case ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>::SLIDING :
            s = "SLIDING";
            break;

        case ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>::STICKY :
            s = "STICKY";
            break;
        }

        std::cout << it->first << " : " << s << std::endl;
        ++it;
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_INL
