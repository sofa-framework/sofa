#ifndef SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL

#include "UnilateralInteractionConstraint.h"

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
    Real dt = delta / dot(P-Q, norm) - dot(Pfree-Qfree, norm);
    if(!(dt>1 && delta >= contactDistance + epsilon))
    {
        int i = contacts.size();
        contacts.resize(i+1);
        Contact& c = contacts[i];
        c.m1 = m1;
        c.m2 = m2;
        c.norm = norm;
        c.delta = delta;
        c.dt = dt;
        c.dfree = dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
        c.friction = friction;
        if (friction) // only if friction, t and s are computed
        {
            c.t = Deriv(norm.y(), norm.z(), norm.x());
            c.s = cross(norm,c.t);
            c.s = c.s / c.s.norm();
            c.t = cross((-norm), c.s);
            c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
            c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);
        }
    }
}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::applyConstraint()
{
    assert(this->object1);
    assert(this->object2);
    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        SparseVecDeriv svd;

        /* Set the constraints vector for the first mechanical model using the normal */
        svd.push_back(SparseDeriv(c.m1, -c.norm));
        c1.push_back(svd);
        /* Set the constraints vector for the second mechanical model using the normal */
        svd[0] = SparseDeriv(c.m2, c.norm);
        c2.push_back(svd);

        if (c.friction)
        {
            /* Set the constraints vector for the first mechanical model using the tangente */
            svd.push_back(SparseDeriv(c.m1, -c.t));
            c1.push_back(svd);
            /* Set the constraints vector for the second mechanical model using the tangente */
            svd[0] = SparseDeriv(c.m2, c.t);
            c2.push_back(svd);

            /* Set the constraints vector for the first mechanical model using the secant */
            svd.push_back(SparseDeriv(c.m1, -c.s));
            c1.push_back(svd);
            /* Set the constraints vector for the second mechanical model the secant */
            svd[0] = SparseDeriv(c.m2, c.s);
            c2.push_back(svd);
        }
    }
}

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::getConstraintValue(double* v, int* offset)
{
    unsigned int i, j;
    j = 0;
    for (i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i]; // get each contact detected
        v[j+(*offset)] = c.delta;
        //		v[j+(*offset)] = c.dfree; // if there is not friction, dfree is the only constraint value added to v
        j++;
        if (c.friction)
        {
            v[j+(*offset)] = c.dfree_t; // if there is friction, dfree_t & dfree_s are added to v too
            v[j+(*offset)] = c.dfree_s;
            j += 2;
        }
    }
    (*offset) += j;
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
