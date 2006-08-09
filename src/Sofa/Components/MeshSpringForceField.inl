#ifndef SOFA_COMPONENTS_MESHSPRINGFORCEFIELD_INL
#define SOFA_COMPONENTS_MESHSPRINGFORCEFIELD_INL

#include "MeshSpringForceField.h"
#include "StiffSpringForceField.inl"
#include "MeshTopology.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void MeshSpringForceField<DataTypes>::addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping)
{
    if (m1<m2)
    {
        if (sset.count(std::make_pair(m1,m2))>0) return;
        sset.insert(std::make_pair(m1,m2));
    }
    else
    {
        if (sset.count(std::make_pair(m2,m1))>0) return;
        sset.insert(std::make_pair(m2,m1));
    }
    Real l = ((*this->object2->getX())[m2] - (*this->object1->getX())[m1]).norm();
    this->springs.push_back(typename SpringForceField<DataTypes>::Spring(m1,m2,stiffness/l, damping/l, l));
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::init()
{
    assert(this->object1);
    assert(this->object2);
    if (this->object1==this->object2)
    {
        MeshTopology* topology = dynamic_cast<MeshTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            std::set< std::pair<int,int> > sset;
            int n;
            Real s, d;
            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
            {
                s = this->linesStiffness;
                d = this->linesDamping;
                n = topology->getNbLines();
                for (int i=0; i<n; ++i)
                {
                    MeshTopology::Line e = topology->getLine(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                }
            }
            if (this->trianglesStiffness != 0.0 || this->trianglesDamping != 0.0)
            {
                s = this->trianglesStiffness;
                d = this->trianglesDamping;
                n = topology->getNbTriangles();
                for (int i=0; i<n; ++i)
                {
                    MeshTopology::Triangle e = topology->getTriangle(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                }
            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                s = this->quadsStiffness;
                d = this->quadsDamping;
                n = topology->getNbQuads();
                for (int i=0; i<n; ++i)
                {
                    MeshTopology::Quad e = topology->getQuad(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }
            if (this->tetrasStiffness != 0.0 || this->tetrasDamping != 0.0)
            {
                s = this->tetrasStiffness;
                d = this->tetrasDamping;
                n = topology->getNbTetras();
                for (int i=0; i<n; ++i)
                {
                    MeshTopology::Tetra e = topology->getTetra(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }
            if (this->cubesStiffness != 0.0 || this->cubesDamping != 0.0)
            {
                s = this->cubesStiffness;
                d = this->cubesDamping;
                n = topology->getNbCubes();
                for (int i=0; i<n; ++i)
                {
                    MeshTopology::Cube e = topology->getCube(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[0], e[4], s, d);
                    this->addSpring(sset, e[0], e[5], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[1], e[4], s, d);
                    this->addSpring(sset, e[1], e[5], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                    this->addSpring(sset, e[2], e[4], s, d);
                    this->addSpring(sset, e[2], e[5], s, d);
                    this->addSpring(sset, e[3], e[4], s, d);
                    this->addSpring(sset, e[3], e[5], s, d);
                    this->addSpring(sset, e[4], e[5], s, d);
                }
            }
        }
    }
}

} // namespace Components

} // namespace Sofa

#endif
