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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_UNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_UNILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraintset/UnilateralInteractionConstraint.h>
#include <sofa/core/behavior/PairInteractionConstraint.inl>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraintset
{
#ifdef SOFA_DEV
template< class DataTypes >
void UnilateralConstraintResolutionWithFriction< DataTypes >::init(int line, double** w, double* force)
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
void UnilateralConstraintResolutionWithFriction< DataTypes >::resolution(int line, double** /*w*/, double* d, double* force, double * /*dfree*/)
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
void UnilateralConstraintResolutionWithFriction< DataTypes >::store(int line, double* force, bool /*convergence*/)
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

#endif // SOFA_DEV

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id, PersistentID localid)
{
    // compute dt and delta
    Real delta = dot(P-Q, norm) - contactDistance;
    Real deltaFree = dot(Pfree-Qfree, norm) - contactDistance;
    Real dt;
    int i = contacts.size();
    contacts.resize(i+1);
    Contact& c = contacts[i];

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

    if (rabs(delta) < 0.00001*ref_dist && rabs(deltaFree) < 0.00001*ref_dist  )
    {

        std::cout<<" case0 "<<std::endl;

        dt=0.0;
        c.dfree = deltaFree;
        c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
        c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);

        return;

    }

    if (rabs(delta - deltaFree) > 0.001 * delta)
    {
        dt = delta / (delta - deltaFree);
        if (dt > 0.0 && dt < 1.0  )
        {
            std::cout<<" case1 : dt = "<<dt<<std::endl;
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
                std::cout<<" case2 "<<std::endl;
                dt=0.0;
                c.dfree = deltaFree; // dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
                //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
                c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
                c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);
            }
            else
            {
                std::cout<<" case3 "<<std::endl;
                dt=1.0;
                c.dfree = deltaFree;
                c.dfree_t = 0;
                c.dfree_s = 0;
            }
        }
    }
    else
    {
        std::cout<<" case4 "<<std::endl;
        dt = 0;
        c.dfree = deltaFree;
        c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
        c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);
        //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
    }


    //sout<<"R_nts = ["<<c.norm<<" ; "<<c.t<<" ; "<<c.s<<" ];"<<sendl;
}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::buildConstraintMatrix(DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &contactId
        , const DataVecCoord &, const DataVecCoord &, const core::ConstraintParams*)
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
            /*
            c1_it.addCol(c.m1, Deriv(1,0,0) * (-c.norm * Deriv(1,0,0)));
            c1_it.addCol(c.m2, Deriv(1,0,0) * (c.norm * Deriv(1,0,0)));
            */
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

//			std::cout << c.norm << std::endl;

            const Deriv u(1,0,0);

//			std::cout << c.norm.linearProduct(u) << std::endl;

            MatrixDerivRowIterator c1_it = c1.writeLine(c.id);
            c1_it.addCol(c.m1, -c.norm);
            //	c1_it.addCol(c.m1, -c.norm.linearProduct(u));

            MatrixDerivRowIterator c2_it = c2.writeLine(c.id);
            //	c2_it.addCol(c.m2, c.norm.linearProduct(u));
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
void UnilateralInteractionConstraint<DataTypes>::getConstraintViolation(defaulttype::BaseVector *v, const DataVecCoord &, const DataVecCoord &
        , const DataVecDeriv &, const DataVecDeriv &, const core::ConstraintParams*)
{
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i]; // get each contact detected

        v->set(c.id, c.dfree);

        if (c.mu > 0.0)
        {
            v->set(c.id+1,c.dfree_t); // dfree_t & dfree_s are added to v to compute the friction
            v->set(c.id+2,c.dfree_s);

            std::cout<<"constraint ["<<i<<"] => dfree = ["<<c.dfree<<" "<<c.dfree_t<<" "<<c.dfree_s<<"]"<<std::endl;
        }
    }
}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::getConstraintInfo(VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& /*positions*/, VecConstDeriv& directions, VecConstArea& /*areas*/)
{
    if (contacts.empty()) return;
    const bool friction = (contacts[0].mu > 0.0); /// @TODO: can there be both friction-less and friction contacts in the same UnilateralInteractionConstraint ???
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

#ifdef SOFA_DEV
template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if(contactsStatus)
        delete[] contactsStatus;
    contactsStatus = new bool[contacts.size()];
    memset(contactsStatus, 0, sizeof(bool)*contacts.size());

    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if(c.mu > 0.0)
        {
//			bool& temp = contactsStatus.at(i);
            resTab[offset] = new UnilateralConstraintResolutionWithFriction<DataTypes>(c.mu, NULL, &contactsStatus[i]);

            // TODO : cette méthode de stockage des forces peu mal fonctionner avec 2 threads quand on utilise l'haptique
//			resTab[offset] = new UnilateralConstraintResolutionWithFriction(c.mu, &prevForces, &contactsStatus[i]);
            offset += 3;
        }
        else
            resTab[offset++] = new UnilateralConstraintResolution();
    }
}

template<class DataTypes>
bool UnilateralInteractionConstraint<DataTypes>::isActive()
{
//	if(!contactsStatus)
    {
        for(unsigned int i=0; i<contacts.size(); i++)
            if(contacts[i].dfree < 0)
                return true;

        return false;
    }
    /*
    	for(unsigned int i=0; i<contacts.size(); i++)
    		if(contactsStatus[i])
    			return true;
    */
    return false;
}
#endif

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor4f(1,0,0,1);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        glLineWidth(1);
        const Contact& c = contacts[i];
#ifdef SOFA_DEV
        if(contactsStatus && contactsStatus[i]) glColor4f(1,0,0,1); else if(c.dfree < 0) glColor4f(1,0,1,1); else
#endif
            glColor4f(1,0.5,0,1);
        helper::gl::glVertexT(c.P);
        helper::gl::glVertexT(c.Q);
        glColor4f(1,1,0,1);
        helper::gl::glVertexT(c.P);
        helper::gl::glVertexT(c.P+c.norm*(c.dfree));
        glColor4f(1,0,1,1);
        helper::gl::glVertexT(c.Q);
        helper::gl::glVertexT(c.Q-c.norm*(c.dfree));

        /*
        if (c.dfree < 0)
        {
        	glLineWidth(5);
        	glColor4f(0,1,0,1);
        	helper::gl::glVertexT(c.Pfree);
        	helper::gl::glVertexT(c.Qfree);
        }
        */
    }
    glEnd();
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

    //sout<<"delta : "<<delta<<" - deltaFree : "<<deltaFree <<sendl;
    //sout<<"P : "<<P<<" - PFree : "<<Pfree <<sendl;
    //sout<<"Q : "<<Q<<" - QFree : "<<Qfree <<sendl;


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

    if (rabs(delta) < 0.00001*ref_dist && rabs(deltaFree) < 0.00001*ref_dist  )
    {

//        std::cout<<" case0 "<<std::endl;

        dt=0.0;
        c.dfree = deltaFree;
        c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
        c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);

        return;

    }

    if (rabs(delta - deltaFree) > 0.001 * delta)
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

#ifdef SOFA_DEV
template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if(contactsStatus)
        delete[] contactsStatus;

    contactsStatus = new bool[contacts.size()];
    memset(contactsStatus, 0, sizeof(bool)*contacts.size());

    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if(c.mu > 0.0)
        {
            UnilateralConstraintResolutionWithFriction<DataTypes> *cRes = new UnilateralConstraintResolutionWithFriction<DataTypes>(c.mu, NULL, &contactsStatus[i]);
            cRes->setConstraint(this);
            resTab[offset] = cRes;
            offset += 3;
        }
        else
            resTab[offset++] = new UnilateralConstraintResolution();
    }
}
#endif


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
        return (contactStates[it->id] == UnilateralConstraintResolutionWithFriction<DataTypes>::STICKY);
    else
        return false;
}

template<class DataTypes>
void ContinuousUnilateralInteractionConstraint<DataTypes>::setContactState(int id, ContactState s)
{
    contactStates.insert(std::make_pair(id, s));
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

    while (it != itEnd)
    {
        std::cout << it->first << " : " << it->second << std::endl;
        ++it;
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
