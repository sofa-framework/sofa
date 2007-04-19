#ifndef SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraint/UnilateralInteractionConstraint.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::addContact(bool friction, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree)
{
    // compute dt and delta
    Real delta = dot(P-Q, norm) - contactDistance;

    // Real dt = delta / dot(P-Q, norm) - dot(Pfree-Qfree, norm);

    Real deltaFree = dot(Pfree-Qfree, norm) - contactDistance;
    Real dt;
    int i = contacts.size();
    contacts.resize(i+1);
    Contact& c = contacts[i];
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.delta = delta;
    c.t = Deriv(norm.z(), norm.x(), norm.y());
    c.s = cross(norm,c.t);
    c.s = c.s / c.s.norm();
    c.t = cross((-norm), c.s);
    c.friction = friction;

    if (rabs(delta - deltaFree) > 0.0001 * delta)
    {
        dt = delta / (delta - deltaFree);
        if (dt < 1.0)
        {
            sofa::defaulttype::Vector3 Qt, Pt;
            Qt = Q*(1-dt) + Qfree*dt;
            Pt = P*(1-dt) + Pfree*dt;
            c.dfree = dot(Pfree-Pt, c.norm) - dot(Qfree-Qt, c.norm);
            c.dfree_t = dot(Pfree-Pt, c.t) - dot(Qfree-Qt, c.t);
            c.dfree_s = dot(Pfree-Pt, c.s) - dot(Qfree-Qt, c.s);
        }
        else
        {
            c.dfree = dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
            c.dfree_t = 0;
            c.dfree_s = 0;
        }
    }
    else
    {
        dt = 0;
        c.dfree = dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
        c.dfree_t = 0;
        c.dfree_s = 0;
    }

}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::applyConstraint(unsigned int &contactId)
{
    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        c.id = contactId++;

        this->object1->setConstraintId(c.id);
        this->object2->setConstraintId(c.id);

        SparseVecDeriv svd1;
        SparseVecDeriv svd2;

        svd1.push_back(SparseDeriv(c.m1, -c.norm));
        svd2.push_back(SparseDeriv(c.m2, c.norm));

        c1.push_back(svd1);
        c2.push_back(svd2);

        /*
        if (c.friction)
        {
        	// Set the constraints vector for the first mechanical model using the tangente
        	svd.push_back(SparseDeriv(c.m1, -c.t));

        	// Set the constraints vector for the second mechanical model using the tangente
        	svd[0] = SparseDeriv(c.m2, c.t);

        	// Set the constraints vector for the first mechanical model using the secant
        	svd.push_back(SparseDeriv(c.m1, -c.s));

        	// Set the constraints vector for the second mechanical model the secant
        	svd[0] = SparseDeriv(c.m2, c.s);
        }
        */
    }
}

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::getConstraintValue(double* v /*, unsigned int &numContacts*/)
{
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i]; // get each contact detected
        v[c.id] = c.dfree; // if there is not friction, dfree is the only constraint value added to v

        //	if (c.friction)
        //	{
        //		v[j+(*offset)] = c.dfree_t; // if there is friction, dfree_t & dfree_s are added to v too
        //		v[j+(*offset)] = c.dfree_s;
        //		j += 2;
        //	}
    }
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif
