/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/topology/RegularGridTopology.h>
//#include <sofa/component/topology/MultiResSparseGridTopology.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/helper/gl/template.h>
#include <GL/gl.h>
#include <algorithm>
#include <iostream>
using std::cerr;
using std::endl;


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::calcMap(topology::MultiResSparseGridTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    int outside = 0;
    typename BarycentricMapping<BasicMapping>::SparseGridMapper* mapper = new typename BarycentricMapping<BasicMapping>::SparseGridMapper(topology);
    this->mapper = mapper;
    mapper->map.resize(out.size());
    int cube;
    for (unsigned int i=0; i<out.size(); i++)
    {
        double coefs[3]= {0,0,0};
        cube = topology->findCube(topology::MultiResSparseGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(topology::MultiResSparseGridTopology::Vec3(out[i]), coefs[0], coefs[1],
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
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::calcMap(topology::RegularGridTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    int outside = 0;
    typename BarycentricMapping<BasicMapping>::RegularGridMapper* mapper = new typename BarycentricMapping<BasicMapping>::RegularGridMapper(topology);
    this->mapper = mapper;
    mapper->map.resize(out.size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        double coefs[3]= {0,0,0};
        int cube = topology->findCube(topology::RegularGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(topology::RegularGridTopology::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::clear()
{
    map1d.clear();
    map2d.clear();
    map3d.clear();
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::addPointInLine(const OutCoord& /*p*/, int lineIndex, const Real* baryCoords)
{
    map1d.resize(map1d.size()+1);
    MappingData<1,0>& data = *map1d.rbegin();
    data.in_index = lineIndex;
    data.baryCoords[0] = baryCoords[0];
    return map1d.size()-1;
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::addPointInTriangle(const OutCoord& /*p*/, int triangleIndex, const Real* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData<2,0>& data = *map2d.rbegin();
    data.in_index = triangleIndex;
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    return map2d.size()-1;
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::addPointInQuad(const OutCoord& /*p*/, int quadIndex, const Real* baryCoords)
{
    map2d.resize(map2d.size()+1);
    MappingData<2,0>& data = *map2d.rbegin();
    data.in_index = quadIndex + topology->getNbTriangles();
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    return map2d.size()-1;
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::addPointInTetra(const OutCoord& /*p*/, int tetraIndex, const Real* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData<3,0>& data = *map3d.rbegin();
    data.in_index = tetraIndex;
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    return map3d.size()-1;
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::addPointInCube(const OutCoord& /*p*/, int cubeIndex, const Real* baryCoords)
{
    map3d.resize(map3d.size()+1);
    MappingData<3,0>& data = *map3d.rbegin();
    data.in_index = cubeIndex + topology->getNbTetras();
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    return map3d.size()-1;
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::createPointInLine(const OutCoord& p, int lineIndex, const InVecCoord* points)
{
    Real baryCoords[1];
    const topology::MeshTopology::Line& elem = topology->getLine(lineIndex);
    const InCoord p0 = (*points)[elem[0]];
    const InCoord pA = (*points)[elem[1]] - p0;
    InCoord pos = p - p0;
    baryCoords[0] = ((pos*pA)/pA.norm2());
    return this->addPointInLine(p, lineIndex, baryCoords);
}

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::createPointInTriangle(const OutCoord& p, int triangleIndex, const InVecCoord* points)
{
    Real baryCoords[2];
    const topology::MeshTopology::Triangle& elem = topology->getTriangle(triangleIndex);
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

template <class BasicMapping>
int BarycentricMapping<BasicMapping>::MeshMapper::createPointInQuad(const OutCoord& p, int quadIndex, const InVecCoord* points)
{
    Real baryCoords[2];
    const topology::MeshTopology::Quad& elem = topology->getQuad(quadIndex);
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::calcMap(topology::MeshTopology* topology)
{
    OutVecCoord& out = *this->toModel->getX();
    InVecCoord& in = *this->fromModel->getX();
    int outside = 0;
    typename BarycentricMapping<BasicMapping>::MeshMapper* mapper = new typename BarycentricMapping<BasicMapping>::MeshMapper(topology);
    this->mapper = mapper;
    mapper->map3d.reserve(out.size());
    const topology::MeshTopology::SeqTetras& tetras = topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = topology->getCubes();
    const topology::MeshTopology::SeqTriangles& triangles = topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = topology->getQuads();
    std::vector<Mat3x3d> bases;
    std::vector<Vec3d> centers;
    if (tetras.empty() && cubes.empty())
    {
        // no 3D elements -> map on 2D elements
        int c0 = triangles.size();
        bases.resize(triangles.size()+quads.size());
        centers.resize(triangles.size()+quads.size());
        for (unsigned int t = 0; t < triangles.size(); t++)
        {
            Mat3x3d m,mt;
            m[0] = in[triangles[t][1]]-in[triangles[t][0]];
            m[1] = in[triangles[t][2]]-in[triangles[t][0]];
            m[2] = cross(m[0],m[1]);
            mt.transpose(m);
            bases[t].invert(mt);
            centers[t] = (in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]])/3;
        }
        for (unsigned int c = 0; c < quads.size(); c++)
        {
            Mat3x3d m,mt;
            m[0] = in[quads[c][1]]-in[quads[c][0]];
            m[1] = in[quads[c][3]]-in[quads[c][0]];
            m[2] = cross(m[0],m[1]);
            mt.transpose(m);
            bases[c0+c].invert(mt);
            centers[c0+c] = (in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]])*0.25;
        }
        for (unsigned int i=0; i<out.size(); i++)
        {
            Vec3d pos = out[i];
            Vec<3,Real> coefs;
            int index = -1;
            double distance = 1e10;
            for (unsigned int t = 0; t < triangles.size(); t++)
            {
                Vec3d v = bases[t] * (pos - in[triangles[t][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max((v[2]<0?-v[2]:v[2])-0.01,v[0]+v[1]-1));
                if (d>0) d = (pos-centers[t]).norm2();
                if (d<distance) { coefs = v; distance = d; index = t; }
            }
            for (unsigned int c = 0; c < quads.size(); c++)
            {
                Vec3d v = bases[c0+c] * (pos - in[quads[c][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(v[1]-1,v[0]-1),std::max(v[2]-0.01,-v[2]-0.01)));
                if (d>0) d = (pos-centers[c0+c]).norm2();
                if (d<distance) { coefs = v; distance = d; index = c0+c; }
            }
            if (distance>0)
            {
                ++outside;
            }
            if (index < c0)
                mapper->addPointInTriangle(pos, index, coefs.ptr());
            else
                mapper->addPointInQuad(pos, index-c0, coefs.ptr());
        }
    }
    else
    {
        int c0 = tetras.size();
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
            Mat3x3d m,mt;
            m[0] = in[cubes[c][1]]-in[cubes[c][0]];
            m[1] = in[cubes[c][2]]-in[cubes[c][0]];
            m[2] = in[cubes[c][4]]-in[cubes[c][0]];
            mt.transpose(m);
            bases[c0+c].invert(mt);
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
        }
    }
    if (outside>0) std::cerr << "WARNING: Barycentric mapping with "<<outside<<"/"<<out.size()<<" points outside of mesh. Can be unstable!"<<std::endl;
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::init()
{
    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    if (topology!=NULL)
    {
#if 0
        topology::MultiResSparseGridTopology* t = dynamic_cast<topology::MultiResSparseGridTopology*>(topology);
        if (t!=NULL)
            this->calcMap(t);
        else
#endif
        {
            topology::RegularGridTopology* t2 = dynamic_cast<topology::RegularGridTopology*>(topology);
            if (t2!=NULL && t2->getNbCubes()>0)
                this->calcMap(t2);
            else
            {
                topology::MeshTopology* t3 = dynamic_cast<topology::MeshTopology*>(topology);
                if (t3!=NULL)
                    this->calcMap(t3);
                else
                {
                    std::cerr << "ERROR: Barycentric mapping does not understand topology."<<std::endl;
                }
            }
        }
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if (mapper!=NULL) mapper->apply(out, in);
}

#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::SparseGridMapper::apply( typename BarycentricMapping<BasicMapping>::Out::VecCoord& out, const typename BarycentricMapping<BasicMapping>::In::VecCoord& in )
{
    out.resize(map.size());
    for(unsigned int i=0; i<map.size(); i++)
    {
        /*if (i == 16)
        	cout << "this->map[i].in_index :" << this->map[i].in_index <<endl;
        */
        const topology::MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);

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
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::RegularGridMapper::apply( typename BarycentricMapping<BasicMapping>::Out::VecCoord& out, const typename BarycentricMapping<BasicMapping>::In::VecCoord& in )
{
    out.resize(map.size());
    for(unsigned int i=0; i<map.size(); i++)
    {
        const topology::RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::apply( typename BarycentricMapping<BasicMapping>::Out::VecCoord& out, const typename BarycentricMapping<BasicMapping>::In::VecCoord& in )
{
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const topology::MeshTopology::SeqLines& lines = this->topology->getLines();
    const topology::MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const topology::MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const topology::MeshTopology::Line& line = lines[index];
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
                const topology::MeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const topology::MeshTopology::Quad& quad = quads[index-c0];
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
                const topology::MeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
                const topology::MeshTopology::Cube& cube = cubes[index-c0];
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(this->toModel->getX()->size());
    if (mapper!=NULL) mapper->applyJ(out, in);
}


#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::SparseGridMapper::applyJ( typename BarycentricMapping<BasicMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BasicMapping>::In::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const topology::MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::RegularGridMapper::applyJ( typename BarycentricMapping<BasicMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BasicMapping>::In::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const topology::RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::applyJ( typename BarycentricMapping<BasicMapping>::Out::VecDeriv& out, const typename BarycentricMapping<BasicMapping>::In::VecDeriv& in )
{
    //cerr<<"	BarycentricMapping<BasicMapping>::MeshMapper::applyJ"<<endl;
    out.resize(map1d.size()+map2d.size()+map3d.size());
    const topology::MeshTopology::SeqLines& lines = this->topology->getLines();
    const topology::MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const topology::MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const topology::MeshTopology::Line& line = lines[index];
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
                const topology::MeshTopology::Triangle& triangle = triangles[index];
                out[i+i0] = in[triangle[0]] * (1-fx-fy)
                        + in[triangle[1]] * fx
                        + in[triangle[2]] * fy;
            }
            else
            {
                const topology::MeshTopology::Quad& quad = quads[index-c0];
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
                const topology::MeshTopology::Tetra& tetra = tetras[index];
                out[i+i0] = in[tetra[0]] * (1-fx-fy-fz)
                        + in[tetra[1]] * fx
                        + in[tetra[2]] * fy
                        + in[tetra[3]] * fz;
            }
            else
            {
                const topology::MeshTopology::Cube& cube = cubes[index-c0];
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (mapper!=NULL) mapper->applyJT(out, in);
}

#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::SparseGridMapper::applyJT( typename BasicMapping::In::VecDeriv& out, const typename BasicMapping::Out::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const topology::MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::RegularGridMapper::applyJT( typename BasicMapping::In::VecDeriv& out, const typename BasicMapping::Out::VecDeriv& in )
{
    for(unsigned int i=0; i<map.size(); i++)
    {
        const OutDeriv v = in[i];
        const topology::RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::applyJT( typename BasicMapping::In::VecDeriv& out, const typename BasicMapping::Out::VecDeriv& in )
{
    const topology::MeshTopology::SeqLines& lines = this->topology->getLines();
    const topology::MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const topology::MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = this->topology->getCubes();
    // 1D elements
    {
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const OutDeriv v = in[i];
            const OutReal fx = (OutReal)map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const topology::MeshTopology::Line& line = lines[index];
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
                const topology::MeshTopology::Triangle& triangle = triangles[index];
                out[triangle[0]] += v * (1-fx-fy);
                out[triangle[1]] += v * fx;
                out[triangle[2]] += v * fy;
            }
            else
            {
                const topology::MeshTopology::Quad& quad = quads[index-c0];
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
                const topology::MeshTopology::Tetra& tetra = tetras[index];
                out[tetra[0]] += v * (1-fx-fy-fz);
                out[tetra[1]] += v * fx;
                out[tetra[2]] += v * fy;
                out[tetra[3]] += v * fz;
            }
            else
            {
                const topology::MeshTopology::Cube& cube = cubes[index-c0];
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

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::draw()
{
    if (!getShow(this)) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    const OutVecCoord& out = *this->toModel->getX();
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<out.size(); i++)
    {
        helper::gl::glVertexT(out[i]);
    }
    glEnd();
    const InVecCoord& in = *this->fromModel->getX();
    if (mapper!=NULL) mapper->draw(out, in);
    glPointSize(1);
}

#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::SparseGridMapper::draw(const typename BasicMapping::Out::VecCoord& out, const typename BasicMapping::In::VecCoord& in)
{
    glBegin (GL_LINES);
    for (unsigned int i=0; i<map.size(); i++)
    {
        const topology::MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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
                helper::gl::glVertexT(out[i]);
                helper::gl::glVertexT(in[cube[j]]);
            }
        }
    }
    glEnd();
}
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::RegularGridMapper::draw(const typename BasicMapping::Out::VecCoord& out, const typename BasicMapping::In::VecCoord& in)
{
    glBegin (GL_LINES);
    for (unsigned int i=0; i<map.size(); i++)
    {
        const topology::RegularGridTopology::Cube cube = this->topology->getCube(this->map[i].in_index);
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
                helper::gl::glVertexT(out[i]);
                helper::gl::glVertexT(in[cube[j]]);
            }
        }
    }
    glEnd();
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::draw(const typename BasicMapping::Out::VecCoord& out, const typename BasicMapping::In::VecCoord& in)
{
    const topology::MeshTopology::SeqLines& lines = this->topology->getLines();
    const topology::MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const topology::MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = this->topology->getCubes();

    glBegin (GL_LINES);
    // 1D elements
    {
        const int i0 = 0;
        for(unsigned int i=0; i<map1d.size(); i++)
        {
            const Real fx = map1d[i].baryCoords[0];
            int index = map1d[i].in_index;
            {
                const topology::MeshTopology::Line& line = lines[index];
                Real f[2];
                f[0] = (1-fx);
                f[1] = fx;
                for (int j=0; j<2; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[line[j]]);
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
                const topology::MeshTopology::Triangle& triangle = triangles[index];
                Real f[3];
                f[0] = (1-fx-fy);
                f[1] = fx;
                f[2] = fy;
                for (int j=0; j<3; j++)
                {
                    if (f[j]<=-0.0001 || f[j]>=0.0001)
                    {
                        glColor3f((float)f[j],1,(float)f[j]);
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[triangle[j]]);
                    }
                }
            }
            else
            {
                const topology::MeshTopology::Quad& quad = quads[index-c0];
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
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[quad[j]]);
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
                const topology::MeshTopology::Tetra& tetra = tetras[index];
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
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[tetra[j]]);
                    }
                }
            }
            else
            {
                const topology::MeshTopology::Cube& cube = cubes[index-c0];
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
                        helper::gl::glVertexT(out[i+i0]);
                        helper::gl::glVertexT(in[cube[j]]);
                    }
                }
            }
        }
    }
    glEnd();
}


/************************************* PropagateConstraint ***********************************/


template <class BasicMapping>
void BarycentricMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    if (mapper!=NULL)
    {
        mapper->applyJT(out, in);
    }
}

#if 0
template <class BasicMapping>
void BarycentricMapping<BasicMapping>::SparseGridMapper::applyJT( typename BasicMapping::In::VecConst& out, const typename BasicMapping::Out::VecConst& in )
{
//	printf("\n applyJT() in BaricentricMapping  [SparseGridMapper] ");
    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            const topology::MultiResSparseGridTopology::Cube cube = this->topology->getCube(this->map[cIn.index].in_index);
            const OutReal fx = (OutReal)map[cIn.index].baryCoords[0];
            const OutReal fy = (OutReal)map[cIn.index].baryCoords[1];
            const OutReal fz = (OutReal)map[cIn.index].baryCoords[2];

            out[i+offset].push_back(InSparseDeriv(cube[0], (InDeriv) (cIn.data * ((1-fx) * (1-fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[1], (InDeriv) (cIn.data * ((  fx) * (1-fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[2], (InDeriv) (cIn.data * ((1-fx) * (  fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[3], (InDeriv) (cIn.data * ((  fx) * (  fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[4], (InDeriv) (cIn.data * ((1-fx) * (1-fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[5], (InDeriv) (cIn.data * ((  fx) * (1-fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[6], (InDeriv) (cIn.data * ((1-fx) * (  fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[7], (InDeriv) (cIn.data * ((  fx) * (  fy) * (  fz)))));
        }
    }
}
#endif

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::RegularGridMapper::applyJT( typename BasicMapping::In::VecConst& out, const typename BasicMapping::Out::VecConst& in )
{
//	printf("\n applyJT() in BaricentricMapping  [RegularGridMapper] ");
    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            const topology::RegularGridTopology::Cube cube = this->topology->getCube(this->map[cIn.index].in_index);
            const OutReal fx = (OutReal)map[cIn.index].baryCoords[0];
            const OutReal fy = (OutReal)map[cIn.index].baryCoords[1];
            const OutReal fz = (OutReal)map[cIn.index].baryCoords[2];

            out[i+offset].push_back(InSparseDeriv(cube[0], (InDeriv) (cIn.data * ((1-fx) * (1-fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[1], (InDeriv) (cIn.data * ((  fx) * (1-fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[2], (InDeriv) (cIn.data * ((1-fx) * (  fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[3], (InDeriv) (cIn.data * ((  fx) * (  fy) * (1-fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[4], (InDeriv) (cIn.data * ((1-fx) * (1-fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[5], (InDeriv) (cIn.data * ((  fx) * (1-fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[6], (InDeriv) (cIn.data * ((1-fx) * (  fy) * (  fz)))));
            out[i+offset].push_back(InSparseDeriv(cube[7], (InDeriv) (cIn.data * ((  fx) * (  fy) * (  fz)))));
        }
    }
}

template <class BasicMapping>
void BarycentricMapping<BasicMapping>::MeshMapper::applyJT( typename BasicMapping::In::VecConst& out, const typename BasicMapping::Out::VecConst& in )
{
//	printf("\n applyJT() in BaricentricMapping  [MeshMapper] \n");
    const topology::MeshTopology::SeqLines& lines = this->topology->getLines();
    const topology::MeshTopology::SeqTriangles& triangles = this->topology->getTriangles();
    const topology::MeshTopology::SeqQuads& quads = this->topology->getQuads();
    const topology::MeshTopology::SeqTetras& tetras = this->topology->getTetras();
    const topology::MeshTopology::SeqCubes& cubes = this->topology->getCubes();

    //const int iLine = lines.size();
    const int iTri = triangles.size();
    //const int iQuad = quads.size();
    const int iTetra= tetras.size();
    //const int iCube = cubes.size();

    const int i1d = map1d.size();
    const int i2d = map2d.size();
    const int i3d = map3d.size();

    int indexIn;

    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            indexIn = cIn.index;
            // 1D elements
            if (indexIn < i1d)
            {
                const OutReal fx = (OutReal)map1d[indexIn].baryCoords[0];
                int index = map1d[indexIn].in_index;
                {
                    const topology::MeshTopology::Line& line = lines[index];
                    out[i+offset].push_back(InSparseDeriv((unsigned) line[0], (InDeriv) cIn.data * (1-fx)));
                    out[i+offset].push_back(InSparseDeriv(line[1], (InDeriv) cIn.data * fx));
                }
            }
            // 2D elements : triangle or quad
            else if (indexIn < i2d)
            {
                const OutReal fx = (OutReal)map2d[indexIn].baryCoords[0];
                const OutReal fy = (OutReal)map2d[indexIn].baryCoords[1];
                int index = map2d[indexIn].in_index;
                if (index < iTri) // triangle
                {
                    const topology::MeshTopology::Triangle& triangle = triangles[index];
                    out[i+offset].push_back(InSparseDeriv(triangle[0], (InDeriv) cIn.data * (1-fx-fy)));
                    out[i+offset].push_back(InSparseDeriv(triangle[1], (InDeriv) cIn.data * fx));
                    out[i+offset].push_back(InSparseDeriv(triangle[2], (InDeriv) cIn.data * fy));
                }
                else // 2D element : Quad
                {
                    const topology::MeshTopology::Quad& quad = quads[index - iTri];
                    out[i+offset].push_back(InSparseDeriv(quad[0], (InDeriv) cIn.data * ((1-fx) * (1-fy))));
                    out[i+offset].push_back(InSparseDeriv(quad[1], (InDeriv) cIn.data * ((  fx) * (1-fy))));
                    out[i+offset].push_back(InSparseDeriv(quad[3], (InDeriv) cIn.data * ((1-fx) * (  fy))));
                    out[i+offset].push_back(InSparseDeriv(quad[2], (InDeriv) cIn.data * ((  fx) * (  fy))));
                }
            }
            // 3D elements
            else if (indexIn < i3d)
            {
                const OutReal fx = (OutReal)map3d[indexIn].baryCoords[0];
                const OutReal fy = (OutReal)map3d[indexIn].baryCoords[1];
                const OutReal fz = (OutReal)map3d[indexIn].baryCoords[2];
                int index = map3d[indexIn].in_index;
                if (index < iTetra) // tetra
                {
                    const topology::MeshTopology::Tetra& tetra = tetras[index];
                    out[i+offset].push_back(InSparseDeriv(tetra[0], (InDeriv) cIn.data * (1-fx-fy-fz)));
                    out[i+offset].push_back(InSparseDeriv(tetra[1], (InDeriv) cIn.data * fx));
                    out[i+offset].push_back(InSparseDeriv(tetra[2], (InDeriv) cIn.data * fy));
                    out[i+offset].push_back(InSparseDeriv(tetra[3], (InDeriv) cIn.data * fz));
                }
                else // cube
                {
                    const topology::MeshTopology::Cube& cube = cubes[index-iTetra];
                    out[i+offset].push_back(InSparseDeriv(cube[0], (InDeriv) cIn.data * ((1-fx) * (1-fy) * (1-fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[1], (InDeriv) cIn.data * ((  fx) * (1-fy) * (1-fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[2], (InDeriv) cIn.data * ((1-fx) * (  fy) * (1-fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[3], (InDeriv) cIn.data * ((  fx) * (  fy) * (1-fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[4], (InDeriv) cIn.data * ((1-fx) * (1-fy) * (  fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[5], (InDeriv) cIn.data * ((  fx) * (1-fy) * (  fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[6], (InDeriv) cIn.data * ((1-fx) * (  fy) * (  fz))));
                    out[i+offset].push_back(InSparseDeriv(cube[7], (InDeriv) cIn.data * ((  fx) * (  fy) * (  fz))));
                }
            }
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
