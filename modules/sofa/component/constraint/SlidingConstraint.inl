#ifndef SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraint/SlidingConstraint.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
void SlidingConstraint<DataTypes>::init()
{
    assert(this->object1);
    assert(this->object2);

    thirdConstraint = 0;
    /*
    m3 = this->object2->getSize();
    this->object2->resize(m3+1);

    m3 = m2b.getValue() + 1;*/
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::applyConstraint(unsigned int &constraintId, double &mu)
{
    mu=0;
    int tm1, tm2a, tm2b;
    tm1 = m1.getValue();
    tm2a = m2a.getValue();
    tm2b = m2b.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    Coord A, B, P, uniAB, dir1, dir2, proj;
    P = (*this->object1->getXfree())[tm1];
    A = (*this->object2->getXfree())[tm2a];
    B = (*this->object2->getXfree())[tm2b];

    // the axis
    uniAB = (B - A);
    Real ab = uniAB.norm();
    uniAB.normalize();

    // projection of the point on the axis
    Real r = (P-A) * uniAB;
    Real r2 = r / ab;
    proj = A + r * uniAB;

    // We move the constraint point onto the projection
    /*	(*this->object2->getX())[m3] = proj;
    	this->getContext()->get<mapping::Mapping>()->updateMapping();

    	return;
    */
    dir1 = P-proj;
    dist = dir1.norm(); // constraint violation
    dir1.normalize(); // direction of the constraint

    dir2 = cross(dir1, uniAB);
    dir2.normalize();

    cid = constraintId;
    constraintId+=2;

    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    this->object1->setConstraintId(cid);
    svd1.push_back(SparseDeriv(tm1, dir1));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.push_back(SparseDeriv(tm2a, -dir1 * (1-r2)));
    svd2.push_back(SparseDeriv(tm2b, -dir1 * r2));
    c2.push_back(svd2);
    svd2.clear();

    /*
    svd2.push_back(SparseDeriv(tm2a, -dir1));
    c2.push_back(svd2);
    svd2[0] = SparseDeriv(tm2b, -dir1);
    //	c2.push_back(svd2);
    */

    /* ajouter un point au mechanical object,
    le déplacer sur la projection du point glissant
    mettre à jour le mapping
    donner la violation de la contrainte
    */

    this->object1->setConstraintId(cid+1);
    svd1[0] = SparseDeriv(tm1, dir2);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2.push_back(SparseDeriv(tm2a, -dir2 * (1-r2)));
    svd2.push_back(SparseDeriv(tm2b, -dir2 * r2));
    c2.push_back(svd2);
    svd2.clear();

    /*	svd2[0] = SparseDeriv(tm2a, -dir2);
    	c2.push_back(svd2);
    	svd2[0] = SparseDeriv(tm2b, -dir2);
    //	c2.push_back(svd2);
    	*/

    thirdConstraint = 0;
    if(r<0)
    {
        thirdConstraint = r;
        constraintId++;

        this->object1->setConstraintId(cid+2);
        svd1[0] = SparseDeriv(tm1, uniAB);
        c1.push_back(svd1);

        this->object2->setConstraintId(cid+2);
        svd2.push_back(SparseDeriv(tm2a, -uniAB));
        c2.push_back(svd2);
    }
    else if(r>ab)
    {
        thirdConstraint = r-ab;
        constraintId++;

        this->object1->setConstraintId(cid+2);
        svd1[0] = SparseDeriv(tm1, -uniAB);
        c1.push_back(svd1);

        this->object2->setConstraintId(cid+2);
        svd2.push_back(SparseDeriv(tm2b, uniAB));
        c2.push_back(svd2);
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintValue(double* v)
{
    v[cid] = dist;
    v[cid+1] = 0.0;

    if(thirdConstraint)
    {
        if(thirdConstraint>0)
            v[cid+2] = -thirdConstraint;
        else
            v[cid+2] = thirdConstraint;
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintId(long* id, unsigned int &offset)
{
    if (!yetIntegrated)
    {
        id[offset++] = -(int)cid;

        yetIntegrated =  true;
    }
    else
    {
        id[offset++] = cid;
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintType(bool* type, unsigned int &offset)
{
    for(int i=0; i<2; i++)
        type[offset++] = true;
    if(thirdConstraint)
        type[offset++] = false;
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    if(thirdConstraint<0)
        glColor4f(1,1,0,1);
    else if(thirdConstraint>0)
        glColor4f(0,1,0,1);
    else
        glColor4f(1,0,1,1);
    helper::gl::glVertexT((*this->object1->getX())[m1.getValue()]);
//	helper::gl::glVertexT((*this->object2->getX())[m3]);
//	helper::gl::glVertexT(proj);
    glEnd();

    glBegin(GL_LINES);
    glColor4f(0,0,1,1);
    helper::gl::glVertexT((*this->object2->getX())[m2a.getValue()]);
    helper::gl::glVertexT((*this->object2->getX())[m2b.getValue()]);
    glEnd();
    glPointSize(1);
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
