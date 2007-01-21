#ifndef SOFA_COMPONENTS_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENTS_BARYCENTRICMAPPING_INL

#include "Sofa/Components/Common/config.h"
#include "Sofa/Components/Common/Mat.h"
#include "BarycentricMapping.h"
#include "RegularGridTopology.h"
#include "MultiResSparseGridTopology.h"

#include "Sofa/Core/MechanicalMapping.inl"
#include "GL/template.h"

#include <GL/gl.h>
#include <algorithm>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::calcMap(MultiResSparseGridTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    int outside = 0;
    typename BarycentricMapping<BaseMapping>::SparseGridMapper* mapper = new typename BarycentricMapping<BaseMapping>::SparseGridMapper(topology);
    this->mapper = mapper;
    mapper->map.resize(out.size());
    int cube;
    for (unsigned int i=0; i<out.size(); i++)
    {
        double coefs[3]= {0,0,0};
        cube = topology->findCube(MultiResSparseGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(MultiResSparseGridTopology::Vec3(out[i]), coefs[0], coefs[1],
                    coefs[2]);
// 			cout << "numero du cube outside : " << cube << endl;
        }
        CubeData& data = mapper->map[i];
        data.baryCoords[0] = (Real)coefs[0];
        data.baryCoords[1] = (Real)coefs[1];
        data.baryCoords[2] = (Real)coefs[2];
        data.in_index = cube;
        //const typename TTopology::Cube points = topology->getCube(cube);
        //std::copy(points.begin(), points.end(), data.points);
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of grid. Can be unstable!"<<std::endl;
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::calcMap(RegularGridTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    int outside = 0;
    typename BarycentricMapping<BaseMapping>::RegularGridMapper* mapper = new typename BarycentricMapping<BaseMapping>::RegularGridMapper(topology);
    this->mapper = mapper;
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
        data.baryCoords[0] = (Real)coefs[0];
        data.baryCoords[1] = (Real)coefs[1];
        data.baryCoords[2] = (Real)coefs[2];
        data.in_index = cube;
        //const typename TTopology::Cube points = topology->getCube(cube);
        //std::copy(points.begin(), points.end(), data.points);
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of grid. Can be unstable!"<<std::endl;
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::clear()
{
    map1d.clear();
    map2d.clear();
    map3d.clear();
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::addPointInLine(const OutCoord& /*p*/, int lineIndex, const Real* baryCoords)
{
    map1d.resize(map1d.size()+1);
    MappingData<1,0>& data = *map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = baryCoords[0];
    return map1d.size()-1;
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::addPointInTriangle(const OutCoord& /*p*/, int triangleIndex, const Real* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData<2,0>& data = *map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    return map2d.size()-1;
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::addPointInQuad(const OutCoord& /*p*/, int quadIndex, const Real* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData<2,0>& data = *map2d.rbegin();
    data.in_index = quadIndex + topology->getNbTriangles();
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    return map2d.size()-1;
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::addPointInTetra(const OutCoord& /*p*/, int tetraIndex, const Real* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData<3,0>& data = *map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    return map3d.size()-1;
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::addPointInCube(const OutCoord& /*p*/, int cubeIndex, const Real* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData<3,0>& data = *map3d.rbegin();
    data.in_index = cubeIndex + topology->getNbTetras();
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    return map3d.size()-1;
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::createPointInLine(const OutCoord& p, int lineIndex, const InVecCoord* points)
{
    Real baryCoords[1];
    const MeshTopology::Line& elem = topology->getLine(lineIndex);
    const InCoord p0 = (*points)[elem[0]];
    const InCoord pA = (*points)[elem[1]] - p0;
    InCoord pos = p - p0;
    baryCoords[0] = ((pos*pA)/pA.norm2());
    return this->addPointInLine(p, lineIndex, baryCoords);
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::createPointInTriangle(const OutCoord& p, int triangleIndex, const InVecCoord* points)
{
    Real baryCoords[2];
    const MeshTopology::Triangle& elem = topology->getTriangle(triangleIndex);
    const InCoord p0 = (*points)[elem[0]];
    const InCoord pA = (*points)[elem[1]] - p0;
    const InCoord pB = (*points)[elem[2]] - p0;
    InCoord pos = p - p0;
    // First project to plane
    InCoord normal = cross(pA, pB);
    Real norm2 = normal.norm2();
    pos -= normal*((pos*normal)/norm2);
    baryCoords[0] = (Real)sqrt(cross(pB, pos).norm2() / norm2);
    baryCoords[1] = (Real)sqrt(cross(pA, pos).norm2() / norm2);
    return this->addPointInTriangle(p, triangleIndex, baryCoords);
}

template <class BaseMapping>
int BarycentricMapping<BaseMapping>::MeshMapper::createPointInQuad(const OutCoord& p, int quadIndex, const InVecCoord* points)
{
    Real baryCoords[2];
    const MeshTopology::Quad& elem = topology->getQuad(quadIndex);
    const InCoord p0 = (*points)[elem[0]];
    const InCoord pA = (*points)[elem[1]] - p0;
    const InCoord pB = (*points)[elem[3]] - p0;
    InCoord pos = p - p0;
    Mat<3,3,typename InCoord::value_type> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross(pA, pB);
    mt.transpose(m);
    base.invert(mt);
    baryCoords[0] = base[0] * pos;
    baryCoords[1] = base[1] * pos;
    return this->addPointInQuad(p, quadIndex, baryCoords);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::calcMap(MeshTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    InVecCoord& in = *this->fromModel->getX();
    int outside = 0;
    typename BarycentricMapping<BaseMapping>::MeshMapper* mapper = new typename BarycentricMapping<BaseMapping>::MeshMapper(topology);
    this->mapper = mapper;
    mapper->map3d.reserve(out.size());
    const MeshTopology::SeqTetras& tetras = topology->getTetras();
    const MeshTopology::SeqCubes& cubes = topology->getCubes();
    int c0 = tetras.size();
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
        m[2] = in[cubes[c][4]]-in[cubes[c][0]];
        bases[c0+c].invert(m);
        centers[c0+c] = (in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]])*0.125;
    }
    for (unsigned int i=0; i<out.size(); i++)
    {
        Vec3d pos = out[i];
        Vec<3,Real> coefs;
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
        if (index < c0)
            mapper->addPointInTetra(pos, index, coefs.ptr());
        else
            mapper->addPointInCube(pos, index-c0, coefs.ptr());
        //MappingData<3,0>& data = mapper->map3d[i];
        //data.baryCoords[0] = (Real)coefs[0];
        //data.baryCoords[1] = (Real)coefs[1];
        //data.baryCoords[2] = (Real)coefs[2];
        //data.in_index = index;
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of mesh. Can be unstable!"<<std::endl;
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::init()
{
    Core::Topology* topology = dynamic_cast<Core::Topology*>(this->fromModel->getContext()->getTopology());
    if (topology!=NULL)
    {
        MultiResSparseGridTopology* t = dynamic_cast<MultiResSparseGridTopology*>(topology);
        if (t!=NULL)
            this->calcMap(t);
        else
        {
            RegularGridTopology* t2 = dynamic_cast<RegularGridTopology*>(topology);
            if (t2!=NULL)
                this->calcMap(t2);
            else
            {
                MeshTopology* t3 = dynamic_cast<MeshTopology*>(topology);
                if (t3!=NULL)
                    this->calcMap(t3);
                else
                {
                    std::cerr << "ERROR: Barycentric mapping does not understand topology."<<std::endl;
                }
            }
        }
    }
    this->BaseMapping::init();
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if (mapper!=NULL) mapper->apply(out, in);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::SparseGridMapper::apply( typename BarycentricMapping<BaseMapping>::Out::VecCoord& out, const typename BarycentricMapping<BaseMapping>::In::VecCoord& in )
{
    out.resize(map.size());
    for(unsigned int i=0; i<map.size(); i++)
    {
        /*if (i == 16)
        	cout << "this->map[i].in_index :" << this->map[i].in_index <<endl;
        */const MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);

        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[7]] * ((  fx) * (  fy) * (  fz));
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::apply( typename BarycentricMapping<BaseMapping>::Out::VecCoord& out, const typename BarycentricMapping<BaseMapping>::In::VecCoord& in )
{
    out.resize(map.size());
    for(unsigned int i=0; i<map.size(); i++)
    {
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[7]] * ((  fx) * (  fy) * (  fz));
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::apply( typename BarycentricMapping<BaseMapping>::Out::VecCoord& out, const typename BarycentricMapping<BaseMapping>::In::VecCoord& in )
{
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const MeshTopology::SeqLines& lines = this->topology->getLines();
    const MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const MeshTopology::Line& line = lines[index];
                out[i] = in[line[0]] * (1-fx)
                        + in[line[1]] * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const MeshTopology::Quad& quad = quads[index-c0];
                out[i+i0] = in[quad[0]] * ((1-fx) * (1-fy))
                        + in[quad[1]] * ((  fx) * (1-fy))
                        + in[quad[3]] * ((1-fx) * (  fy))
                        + in[quad[2]] * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
                const MeshTopology::Cube& cube = cubes[index-c0];
                out[i+i0] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                        + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                        + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                        + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                        + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                        + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[7]] * ((  fx) * (  fy) * (  fz));
            }
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
void BarycentricMapping<BaseMapping>::SparseGridMapper::applyJ( typename BarycentricMapping<BaseMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BaseMapping>::In::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[7]] * ((  fx) * (  fy) * (  fz));
    }
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
        out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                + in[cube[7]] * ((  fx) * (  fy) * (  fz));
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::applyJ( typename BarycentricMapping<BaseMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BaseMapping>::In::VecDeriv& in )
{
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const MeshTopology::SeqLines& lines = this->topology->getLines();
    const MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const MeshTopology::Line& line = lines[index];
                out[i] = in[line[0]] * (1-fx)
                        + in[line[1]] * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const MeshTopology::Quad& quad = quads[index-c0];
                out[i+i0] = in[quad[0]] * ((1-fx) * (1-fy))
                        + in[quad[1]] * ((  fx) * (1-fy))
                        + in[quad[3]] * ((1-fx) * (  fy))
                        + in[quad[2]] * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
                const MeshTopology::Cube& cube = cubes[index-c0];
                out[i+i0] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
                        + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
                        + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
                        + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
                        + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
                        + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
                        + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
                        + in[cube[7]] * ((  fx) * (  fy) * (  fz));
            }
        }
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (mapper!=NULL) mapper->applyJT(out, in);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::SparseGridMapper::applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        const OutReal fz = (OutReal)map[i].baryCoords[2];
        out[cube[0]] += v * ((1-fx) * (1-fy) * (1-fz));
        out[cube[1]] += v * ((  fx) * (1-fy) * (1-fz));
        out[cube[2]] += v * ((1-fx) * (  fy) * (1-fz));
        out[cube[3]] += v * ((  fx) * (  fy) * (1-fz));
        out[cube[4]] += v * ((1-fx) * (1-fy) * (  fz));
        out[cube[5]] += v * ((  fx) * (1-fy) * (  fz));
        out[cube[6]] += v * ((1-fx) * (  fy) * (  fz));
        out[cube[7]] += v * ((  fx) * (  fy) * (  fz));
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::RegularGridMapper::applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
        const OutReal fx = (OutReal)map[i].baryCoords[0];
        const OutReal fy = (OutReal)map[i].baryCoords[1];
        const OutReal fz = (OutReal)map[i].baryCoords[2];
        out[cube[0]] += v * ((1-fx) * (1-fy) * (1-fz));
        out[cube[1]] += v * ((  fx) * (1-fy) * (1-fz));
        out[cube[2]] += v * ((1-fx) * (  fy) * (1-fz));
        out[cube[3]] += v * ((  fx) * (  fy) * (1-fz));
        out[cube[4]] += v * ((1-fx) * (1-fy) * (  fz));
        out[cube[5]] += v * ((  fx) * (1-fy) * (  fz));
        out[cube[6]] += v * ((1-fx) * (  fy) * (  fz));
        out[cube[7]] += v * ((  fx) * (  fy) * (  fz));
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in )
{
    const MeshTopology::SeqLines& lines = this->topology->getLines();
    const MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const OutDeriv v = in[i];
            const OutReal fx = (OutReal)map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const MeshTopology::Line& line = lines[index];
                out[line[0]] += v * (1-fx);
                out[line[1]] += v * fx;
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const OutDeriv v = in[i+i0];
            const OutReal fx = (OutReal)map2d[i].baryCoords[0];
            const OutReal fy = (OutReal)map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Triangle& triangle = triangles[index];
                out[triangle[0]] += v * (1-fx-fy);
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
            }
            else
            {
                const MeshTopology::Quad& quad = quads[index-c0];
                out[quad[0]] += v * ((1-fx) * (1-fy));
                out[quad[1]] += v * ((  fx) * (1-fy));
                out[quad[3]] += v * ((1-fx) * (  fy));
                out[quad[2]] += v * ((  fx) * (  fy));
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size() + map2d.size();
        const int c0 = tetras.size();
        for(unsigned int i=0; i<map3d.size(); i++)
        {
            const OutDeriv v = in[i+i0];
            const OutReal fx = (OutReal)map3d[i].baryCoords[0];
            const OutReal fy = (OutReal)map3d[i].baryCoords[1];
            const OutReal fz = (OutReal)map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
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
                out[cube[0]] += v * ((1-fx) * (1-fy) * (1-fz));
                out[cube[1]] += v * ((  fx) * (1-fy) * (1-fz));
                out[cube[2]] += v * ((1-fx) * (  fy) * (1-fz));
                out[cube[3]] += v * ((  fx) * (  fy) * (1-fz));
                out[cube[4]] += v * ((1-fx) * (1-fy) * (  fz));
                out[cube[5]] += v * ((  fx) * (1-fy) * (  fz));
                out[cube[6]] += v * ((1-fx) * (  fy) * (  fz));
                out[cube[7]] += v * ((  fx) * (  fy) * (  fz));
            }
        }
    }
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::draw()
{
    if (!getShow(this)) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    const OutVecCoord& out = *this->toModel->getX();
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<out.size(); i++)
    {
        GL::glVertexT(out[i]);
    }
    glEnd();
    const InVecCoord& in = *this->fromModel->getX();
    if (mapper!=NULL) mapper->draw(out, in);
    glPointSize(1);
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::SparseGridMapper::draw(const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in)
{
    glBegin (GL_LINES);
    for (unsigned int i=0; i<map.size(); i++)
    {
        const MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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
                glColor3f((float)f[j],(float)f[j],1);
                GL::glVertexT(out[i]);
                GL::glVertexT(in[cube[j]]);
            }
        }
    }
    glEnd();
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
                glColor3f((float)f[j],(float)f[j],1);
                GL::glVertexT(out[i]);
                GL::glVertexT(in[cube[j]]);
            }
        }
    }
    glEnd();
}

template <class BaseMapping>
void BarycentricMapping<BaseMapping>::MeshMapper::draw(const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in)
{
    const MeshTopology::SeqLines& lines = this->topology->getLines();
    const MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const MeshTopology::SeqCubes& cubes = this->topology->getCubes();

    glBegin (GL_LINES);
    // 1D elements
    {
        const int i0 = 0;
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const MeshTopology::Line& line = lines[index];
                Real f[2];
                f[0] = (1-fx);
                f[1] = fx;
                for (int j=0; j<2; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        GL::glVertexT(out[i+i0]);
                        GL::glVertexT(in[line[j]]);
                    }
                }
            }
        }
    }
    // 2D elements
    {
        const int i0 = map1d.size();
        const int c0 = triangles.size();
        for(unsigned int i=0; i<map2d.size(); i++)
        {
            const Real fx = map2d[i].baryCoords[0];
            const Real fy = map2d[i].baryCoords[1];
            int index = map2d[i].in_index;
            if (index<c0)
            {
                const MeshTopology::Triangle& triangle = triangles[index];
                Real f[3];
                f[0] = (1-fx-fy);
                f[1] = fx;
                f[2] = fy;
                for (int j=0; j<3; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        GL::glVertexT(out[i+i0]);
                        GL::glVertexT(in[triangle[j]]);
                    }
                }
            }
            else
            {
                const MeshTopology::Quad& quad = quads[index-c0];
                Real f[4];
                f[0] = ((1-fx) * (1-fy));
                f[1] = ((  fx) * (1-fy));
                f[3] = ((1-fx) * (  fy));
                f[2] = ((  fx) * (  fy));
                for (int j=0; j<4; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        GL::glVertexT(out[i+i0]);
                        GL::glVertexT(in[quad[j]]);
                    }
                }
            }
        }
    }
    // 3D elements
    {
        const int i0 = map1d.size()+map2d.size();
        const int c0 = tetras.size();
        for (unsigned int i=0; i<map3d.size(); i++)
        {
            const Real fx = map3d[i].baryCoords[0];
            const Real fy = map3d[i].baryCoords[1];
            const Real fz = map3d[i].baryCoords[2];
            int index = map3d[i].in_index;
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
                        glColor3f((float)f[j],1,(float)f[j]);
                        GL::glVertexT(out[i+i0]);
                        GL::glVertexT(in[tetra[j]]);
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
                        glColor3f((float)f[j],1,1);
                        GL::glVertexT(out[i+i0]);
                        GL::glVertexT(in[cube[j]]);
                    }
                }
            }
        }
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
