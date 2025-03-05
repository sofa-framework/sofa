/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/io/mesh/MeshXspLoader.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/io/XspLoader.h>
using sofa::helper::io::XspLoader;
using sofa::helper::io::XspLoaderDataHook;

using sofa::helper::WriteOnlyAccessor;

#include <sofa/type/Vec.h>
using sofa::type::Vec3;

#include <sofa/core/topology/Topology.h>
using sofa::core::topology::Topology;

namespace sofa::component::io::mesh
{

class MeshXspLoadDataHook : public XspLoaderDataHook
{
public:
    MeshXspLoader* m_data;
    WriteOnlyAccessor<decltype(m_data->d_positions)> m_positions;
    WriteOnlyAccessor<decltype(m_data->d_edges)> m_edges;

    MeshXspLoadDataHook(MeshXspLoader* data);
    ~MeshXspLoadDataHook() override;

    void setNumMasses(size_t n) override { m_positions.reserve(n); }
    void setNumSprings(size_t n) override { m_edges.reserve(n); }

    void finalizeLoading(bool isOk) override
    {
        if(!isOk){
            m_positions.clear();
            m_edges.clear();
        }
    }

    void addMass(SReal px, SReal py, SReal pz, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) override
    {
        m_positions.push_back(Vec3(px,py,pz));
    }

    void addSpring(size_t index1, size_t index2, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) override
    {
        m_edges.push_back(Topology::Edge(index1, index2));
    }

    void addVectorSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos, SReal /*restx*/, SReal /*resty*/, SReal /*restz*/) override
    {
        addSpring(m1, m2, ks, kd, initpos);
    }
};

MeshXspLoadDataHook::MeshXspLoadDataHook(MeshXspLoader* data) :
    m_data(data),
    m_positions(m_data->d_positions),
    m_edges(m_data->d_edges)
{}

MeshXspLoadDataHook::~MeshXspLoadDataHook() {}



MeshXspLoader::MeshXspLoader() : MeshLoader() {}

bool MeshXspLoader::doLoad()
{
    MeshXspLoadDataHook data(this);
    return XspLoader::Load(d_filename.getValue(), data);
}


void MeshXspLoader::doClearBuffers()
{
    /// Nothing to do if no output is added to the "filename" dataTrackerEngine.
}

void registerMeshXspLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Specific mesh loader for Xsp file format.")
        .add< MeshXspLoader >());
}

} //namespace sofa::component::io::mesh
