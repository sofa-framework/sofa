#ifndef SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraint/BilateralInteractionConstraint.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::init()
{
    assert(this->object1);
    assert(this->object2);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::applyConstraint(unsigned int &constraintId, double &mu)
{
    mu=0;
    int tm1, tm2;
    tm1 = m1.getValue();
    tm2 = m2.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    Coord cx(1,0,0), cy(0,1,0), cz(0,0,1);

    cid = constraintId;
    constraintId+=3;

    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    this->object1->setConstraintId(cid);
    svd1.push_back(SparseDeriv(tm1, -cx));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.push_back(SparseDeriv(tm2, cx));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+1);
    svd1[0] = SparseDeriv(tm1, -cy);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2[0] = SparseDeriv(tm2, cy);
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+2);
    svd1[0] = SparseDeriv(tm1, -cz);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+2);
    svd2[0] = SparseDeriv(tm2, cz);
    c2.push_back(svd2);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintValue(double* v)
{
    dfree = (*this->object2->getXfree())[m2.getValue()] - (*this->object1->getXfree())[m1.getValue()];

    v[cid] = dfree[0];
    v[cid+1] = dfree[1];
    v[cid+2] = dfree[2];
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintId(long* id, unsigned int &offset)
{
    if (!yetIntegrated)
    {
        id[offset++] = -(int)cid;

        yetIntegrated = true;
    }
    else
    {
        id[offset++] = cid;
    }
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::componentmodel::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    for(int i=0; i<3; i++)
        resTab[offset++] = new BilateralConstraintResolution();
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    glColor4f(1,0,1,1);
    helper::gl::glVertexT((*this->object1->getX())[m1.getValue()]);
    helper::gl::glVertexT((*this->object2->getX())[m2.getValue()]);
    glEnd();
    glPointSize(1);
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
