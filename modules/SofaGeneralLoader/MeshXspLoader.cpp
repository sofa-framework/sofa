/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaGeneralLoader/MeshXspLoader.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/io/MassSpringLoader.h>
using sofa::helper::io::XspLoader;
using sofa::helper::io::XspLoaderDataHook;

using sofa::helper::WriteOnlyAccessor;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3;

#include <sofa/core/topology/Topology.h>
using sofa::core::topology::Topology;

namespace sofa
{

namespace component
{

namespace loader
{

class MeshXspLoadDataHook : public XspLoaderDataHook
{
public:
    MeshXspLoader* m_data;
    WriteOnlyAccessor<decltype(m_data->d_positions)> m_positions;
    WriteOnlyAccessor<decltype(m_data->d_edges)> m_edges;

    MeshXspLoadDataHook(MeshXspLoader* data) :
        m_data(data),
        m_positions(m_data->d_positions),
        m_edges(m_data->d_edges)
    {}

    virtual ~MeshXspLoadDataHook(){}

    void setNumMasses(size_t n) override { m_positions.reserve(n); }
    void setNumSprings(size_t n) override { m_edges.reserve(n); }

    void addMass(SReal px, SReal py, SReal pz, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/)
    {
        m_positions.push_back(Vec3(px,py,pz));
    }

    void addSpring(int index1, int index2, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) override
    {
        m_edges.push_back(Topology::Edge(index1, index2));
    }

    void addVectorSpring(int m1, int m2, SReal ks, SReal kd, SReal initpos, SReal /*restx*/, SReal /*resty*/, SReal /*restz*/) override
    {
        addSpring(m1, m2, ks, kd, initpos);
    }
};

MeshXspLoader::MeshXspLoader() : MeshLoader() {}

bool MeshXspLoader::load()
{
    MeshXspLoadDataHook data(this);
    return XspLoader::Load(m_filename.getValue(), data, this);
}

static int MeshXspLoaderClass = core::RegisterObject("Specific mesh loader for Xsp file format.")
        .add< MeshXspLoader >();
} /// namespace loader

} /// namespace component

} /// namespace sofa

