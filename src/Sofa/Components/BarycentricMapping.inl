#ifndef SOFA_COMPONENTS_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENTS_BARYCENTRICMAPPING_INL

#include "Sofa/Components/Common/config.h"
#include "Sofa/Components/Common/Mat.h"
#include "BarycentricMapping.h"
#include "RegularGridTopology.h"
#include "Scene.h"

#include "Sofa/Core/MechanicalMapping.inl"

#include <GL/gl.h>
#include <algorithm>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::calcMap(RegularGridTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    int outside = 0;
    typename BarycentricMapping<BaseMapping>::RegularGridMapper* mapper = new typename BarycentricMapping<BaseMapping>::RegularGridMapper;
    this->mapper = mapper;
    mapper->topology = topology;
    mapper->map.resize(out.size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        double coefs[3]= {0,0,0};
        int cube = topology->findCube(RegularGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(RegularGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        }
        CubeData& data = mapper->map[i];
        std::copy(coefs, coefs+3, data.baryCoords);
        data.in_index = cube;
        //const typename TTopology::Cube points = topology->getCube(cube);
        //std::copy(points.begin(), points.end(), data.points);
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of grid. Can be unstable!"<<std::endl;
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::calcMap(MeshTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    InVecCoord& in = *this->fromModel->getX();
    int outside = 0;
    typename BarycentricMapping<BaseMapping>::MeshMapper* mapper = new typename BarycentricMapping<BaseMapping>::MeshMapper;
    this->mapper = mapper;
    mapper->topology = topology;
    mapper->map.resize(out.size());
    const MeshTopology::SeqTetras& tetras = topology->getTetras();
    const MeshTopology::SeqCubes& cubes = topology->getCubes();
    unsigned int c0 = tetras.size();
    std::vector<Mat3x3d> bases;
    std::vector<Vec3d> centers;
    bases.resize(tetras.size()+cubes.size());
    centers.resize(tetras.size()+cubes.size());
    for (unsigned int t = 0; t < tetras.size(); t++)
    {
        Mat3x3d m,mt;
        m[0] = in[tetras[t][1]]-in[tetras[t][0]];
        m[1] = in[tetras[t][2]]-in[tetras[t][0]];
        m[2] = in[tetras[t][3]]-in[tetras[t][0]];
        mt.transpose(m);
        bases[t].invert(mt);
        centers[t] = (in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]])*0.25;
        //std::cout << "Tetra "<<t<<" center="<<centers[t]<<" base="<<m<<std::endl;
    }
    for (unsigned int c = 0; c < cubes.size(); c++)
    {
        Mat3x3d m;
        m[0] = in[cubes[c][1]]-in[cubes[c][0]];
        m[1] = in[cubes[c][2]]-in[cubes[c][0]];
        m[2] = in[cubes[c][3]]-in[cubes[c][0]];
        bases[c0+c].invert(m);
        centers[c0+c] = (in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]])*0.125;
    }
    for (unsigned int i=0; i<out.size(); i++)
    {
        Vec3d pos = out[i];
        Vec3d coefs;
        int index = -1;
        double distance = 1e10;
        for (unsigned int t = 0; t < tetras.size(); t++)
        {
            Vec3d v = bases[t] * (pos - in[tetras[t][0]]);
            double d = std::max(std::max(-v[0],-v[1]),std::max(-v[2],v[0]+v[1]+v[2]-1));
            if (d>0) d = (pos-centers[t]).norm2();
            if (d<distance) { coefs = v; distance = d; index = t; }
        }
        for (unsigned int c = 0; c < cubes.size(); c++)
        {
            Vec3d v = bases[c0+c] * (pos - in[cubes[c][0]]);
            double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(-v[2],v[0]-1),std::max(v[1]-1,v[2]-1)));
            if (d>0) d = (pos-centers[c0+c]).norm2();
            if (d<distance) { coefs = v; distance = d; index = c0+c; }
        }
        if (distance>0)
        {
            ++outside;
        }
        MappingData<3,0>& data = mapper->map[i];
        std::copy(coefs.begin(), coefs.end(), data.baryCoords);
        data.in_index = index;
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of mesh. Can be unstable!"<<std::endl;
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::init()
{
    Core::Topology* topology = this->fromModel->getTopology();
    if (topology!=NULL)
    {
        RegularGridTopology* t = dynamic_cast<RegularGridTopology*>(topology);
        if (t!=NULL)
            this->calcMap(t);
        else
        {
            MeshTopology* t2 = dynamic_cast<MeshTopology*>(topology);
            if (t2!=NULL)
                this->calcMap(t2);
            else
            {
                std::cerr << "ERROR: Barycentric mapping does not understand topology."<<std::endl;
            }
        }
    }
    this->BaseMapping::init();
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    //out.resize(map.size());
    if (mapper!=NULL) mapper->apply(out, in);
}


template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::apply( typename BarycentricMapping<BaseMapping>::Out::VecCoord& out, const typename BarycentricMapping<BaseMapping>::In::VecCoord& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[i] = in[cube[0]] * (1-fx) * (1-fy) * (1-fz)
                + in[cube[1]] * (  fx) * (1-fy) * (1-fz)
                + in[cube[2]] * (1-fx) * (  fy) * (1-fz)
                + in[cube[3]] * (  fx) * (  fy) * (1-fz)
                + in[cube[4]] * (1-fx) * (1-fy) * (  fz)
                + in[cube[5]] * (  fx) * (1-fy) * (  fz)
                + in[cube[6]] * (1-fx) * (  fy) * (  fz)
                + in[cube[7]] * (  fx) * (  fy) * (  fz);
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::apply( typename BarycentricMapping<BaseMapping>::Out::VecCoord& out, const typename BarycentricMapping<BaseMapping>::In::VecCoord& in )
{
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    int c0 = tetras.size();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        if (index<c0)
        {
            const MeshTopology::Tetra& tetra = tetras[index];
            out[i] = in[tetra[0]] * (1-fx-fy-fz)
                    + in[tetra[1]] * fx
                    + in[tetra[2]] * fy
                    + in[tetra[3]] * fz;
        }
        else
        {
            const MeshTopology::Cube& cube = cubes[index-c0];
            out[i] = in[cube[0]] * (1-fx) * (1-fy) * (1-fz)
                    + in[cube[1]] * (  fx) * (1-fy) * (1-fz)
                    + in[cube[2]] * (1-fx) * (  fy) * (1-fz)
                    + in[cube[3]] * (  fx) * (  fy) * (1-fz)
                    + in[cube[4]] * (1-fx) * (1-fy) * (  fz)
                    + in[cube[5]] * (  fx) * (1-fy) * (  fz)
                    + in[cube[6]] * (1-fx) * (  fy) * (  fz)
                    + in[cube[7]] * (  fx) * (  fy) * (  fz);
        }
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(this->toModel->getX()->size());
    if (mapper!=NULL) mapper->applyJ(out, in);
}


template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::applyJ( typename BarycentricMapping<BaseMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BaseMapping>::In::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[i] = in[cube[0]] * (1-fx) * (1-fy) * (1-fz)
                + in[cube[1]] * (  fx) * (1-fy) * (1-fz)
                + in[cube[2]] * (1-fx) * (  fy) * (1-fz)
                + in[cube[3]] * (  fx) * (  fy) * (1-fz)
                + in[cube[4]] * (1-fx) * (1-fy) * (  fz)
                + in[cube[5]] * (  fx) * (1-fy) * (  fz)
                + in[cube[6]] * (1-fx) * (  fy) * (  fz)
                + in[cube[7]] * (  fx) * (  fy) * (  fz);
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::applyJ( typename BarycentricMapping<BaseMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BaseMapping>::In::VecDeriv& in )
{
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    int c0 = tetras.size();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        if (index<c0)
        {
            const MeshTopology::Tetra& tetra = tetras[index];
            out[i] = in[tetra[0]] * (1-fx-fy-fz)
                    + in[tetra[1]] * fx
                    + in[tetra[2]] * fy
                    + in[tetra[3]] * fz;
        }
        else
        {
            const MeshTopology::Cube& cube = cubes[index-c0];
            out[i] = in[cube[0]] * (1-fx) * (1-fy) * (1-fz)
                    + in[cube[1]] * (  fx) * (1-fy) * (1-fz)
                    + in[cube[2]] * (1-fx) * (  fy) * (1-fz)
                    + in[cube[3]] * (  fx) * (  fy) * (1-fz)
                    + in[cube[4]] * (1-fx) * (1-fy) * (  fz)
                    + in[cube[5]] * (  fx) * (1-fy) * (  fz)
                    + in[cube[6]] * (1-fx) * (  fy) * (  fz)
                    + in[cube[7]] * (  fx) * (  fy) * (  fz);
        }
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (mapper!=NULL) mapper->applyJT(out, in);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[cube[0]] += v * (1-fx) * (1-fy) * (1-fz);
        out[cube[1]] += v * (  fx) * (1-fy) * (1-fz);
        out[cube[2]] += v * (1-fx) * (  fy) * (1-fz);
        out[cube[3]] += v * (  fx) * (  fy) * (1-fz);
        out[cube[4]] += v * (1-fx) * (1-fy) * (  fz);
        out[cube[5]] += v * (  fx) * (1-fy) * (  fz);
        out[cube[6]] += v * (1-fx) * (  fy) * (  fz);
        out[cube[7]] += v * (  fx) * (  fy) * (  fz);
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in )
{
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    int c0 = tetras.size();
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        if (index<c0)
        {
            const MeshTopology::Tetra& tetra = tetras[index];
            out[tetra[0]] += v * (1-fx-fy-fz);
            out[tetra[1]] += v * fx;
            out[tetra[2]] += v * fy;
            out[tetra[3]] += v * fz;
        }
        else
        {
            const MeshTopology::Cube& cube = cubes[index-c0];
            out[cube[0]] += v * (1-fx) * (1-fy) * (1-fz);
            out[cube[1]] += v * (  fx) * (1-fy) * (1-fz);
            out[cube[2]] += v * (1-fx) * (  fy) * (1-fz);
            out[cube[3]] += v * (  fx) * (  fy) * (1-fz);
            out[cube[4]] += v * (1-fx) * (1-fy) * (  fz);
            out[cube[5]] += v * (  fx) * (1-fy) * (  fz);
            out[cube[6]] += v * (1-fx) * (  fy) * (  fz);
            out[cube[7]] += v * (  fx) * (  fy) * (  fz);
        }
    }
}

static inline void glVertex3v(const float* p) { glVertex3fv(p); }
static inline void glVertex3v(const double* p) { glVertex3dv(p); }

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::draw()
{
    if (!Scene::getInstance()->getShowMappings()) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    const OutVecCoord& out = *this->toModel->getX();
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<out.size(); i++)
    {
        glVertex3v(out[i]);
    }
    glEnd();
    InVecCoord& in = *this->fromModel->getX();
    if (mapper!=NULL) mapper->draw(out, in);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::draw(const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in)
{
    glBegin (GL_LINES);
    for (unsigned int i=0; i<map.size(); i++)
    {
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        Real f[8];
        f[0] = (1-fx) * (1-fy) * (1-fz);
        f[1] = (  fx) * (1-fy) * (1-fz);
        f[2] = (1-fx) * (  fy) * (1-fz);
        f[3] = (  fx) * (  fy) * (1-fz);
        f[4] = (1-fx) * (1-fy) * (  fz);
        f[5] = (  fx) * (1-fy) * (  fz);
        f[6] = (1-fx) * (  fy) * (  fz);
        f[7] = (  fx) * (  fy) * (  fz);
        for (int j=0; j<8; j++)
        {
            if (f[j]<=-0.0001 || f[j]>=0.0001)
            {
                glColor3f(f[j],f[j],1);
                glVertex3v(out[i]); glVertex3v(in[cube[j]]);
            }
        }
    }
    glEnd();
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::draw(const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in)
{
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    int c0 = tetras.size();
    glBegin (GL_LINES);
    for (unsigned int i=0; i<map.size(); i++)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        int index = map[i].in_index;
        if (index<c0)
        {
            const MeshTopology::Tetra& tetra = tetras[index];
            Real f[4];
            f[0] = (1-fx-fy-fz);
            f[1] = fx;
            f[2] = fy;
            f[3] = fz;
            for (int j=0; j<4; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f(f[j],1,f[j]);
                    glVertex3v(out[i]); glVertex3v(in[tetra[j]]);
                }
            }
        }
        else
        {
            const MeshTopology::Cube& cube = cubes[index-c0];
            Real f[8];
            f[0] = (1-fx) * (1-fy) * (1-fz);
            f[1] = (  fx) * (1-fy) * (1-fz);
            f[2] = (1-fx) * (  fy) * (1-fz);
            f[3] = (  fx) * (  fy) * (1-fz);
            f[4] = (1-fx) * (1-fy) * (  fz);
            f[5] = (  fx) * (1-fy) * (  fz);
            f[6] = (1-fx) * (  fy) * (  fz);
            f[7] = (  fx) * (  fy) * (  fz);
            for (int j=0; j<8; j++)
            {
                if (f[j]<=-0.0001 || f[j]>=0.0001)
                {
                    glColor3f(f[j],1,1);
                    glVertex3v(out[i]); glVertex3v(in[cube[j]]);
                }
            }
        }
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
