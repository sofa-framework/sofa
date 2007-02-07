#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/PlaneForceField.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <GL/gl.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

//---- Added by Xunlei Wu ----
//The contact forces are computed as following:
//1. When a Tetrahedron node penetrates the surface, the penetration vector
//   is converted to tetrahedron mesh's nodal local initial frame by
//   applying the inverse of current nodal Quaternion on this vector;
//2. Assemble these 3-by-1 vectors into a global vector and multiply it on
//   the left with globalInitialComplianceMatrix.
//3. The result vector is the force vector in the mesh nodal initial frame.
//   Convert these force vectors back to the global frame by current nodal
//   Quaternions before adding back to "f1".
template<class DataTypes>
void PlaneForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& /*v1*/)
{
    assert(this->object);
    assert(this->fem);

    const std::vector<Quaternion>& Qs = this->fem->getNodalQuaternions();
    const NewMAT::SymmetricMatrix& C = this->fem->getGlobalInitialComplianceMatrix();
    //const NewMAT::SymmetricMatrix& K = this->fem->getGlobalInitialStiffnessMatrix();

    static NewMAT::ColumnVector assembledV(p1.size() * 3);
    static NewMAT::ColumnVector assembledF(p1.size() * 3);

    _force.resize(p1.size());

    this->contacts.clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        unsigned int i3 = 3 * i;
        Real d = p1[i]*planeNormal-planeD;
        if (d < 0)
        {
            this->contacts.push_back(i);

            Vec<3, Real> localV, globalV;
            globalV = planeNormal * (-d);
            localV  = Qs[i].inverseRotate(globalV);
            assembledV(i3+1) = (Real)localV[0];
            assembledV(i3+2) = (Real)localV[1];
            assembledV(i3+3) = (Real)localV[2];
        }
        else
        {
            assembledV(i3+1) = assembledV(i3+2) = assembledV(i3+3) = 0;
        }
    }
    assembledF = C * assembledV;
    Vec<3, Real> gravity(0, -9.8, 0);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        unsigned int i3 = 3 * i;
        Vec<3, Real> localF, globalF;
        localF[0] = (Real)assembledF(i3+1);
        localF[1] = (Real)assembledF(i3+2);
        localF[2] = (Real)assembledF(i3+3);
        //globalF = Qs[i].Rotate(localF);
        _force[i] = Qs[i].rotate(localF);
        Real d = p1[i]*planeNormal-planeD;
        if (d < 0)
            f1[i] += _force[i] - gravity;
        else
            f1[i] += _force[i];
    }
}
// -------------------------------------------------------------------


template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce(VecDeriv& f1,  const VecDeriv& dx1)
{
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        f1[p] += planeNormal * (-this->stiffness * (dx1[p]*planeNormal));
    }
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    VecCoord& p1 = *this->mstate->getX();
    glDisable(GL_LIGHTING);
    glColor4f(1,0,0,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        Coord p2 = p1[i];
        p2 += planeNormal*(-d);
        if (d<0)
        {
            glVertex3d(p1[i][0],p1[i][1],p1[i][2]);
            glVertex3d(p2[0],p2[1],p2[2]);
        }
    }
    glEnd();
    glPointSize(1);
    glColor4f(0,1,0,1);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        Coord p2 = p1[i];
        p2 += planeNormal*(-d);
        if (d>=0)
            glVertex3d(p2[0],p2[1],p2[2]);
    }
    glEnd();
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
