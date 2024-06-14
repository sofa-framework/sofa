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
#include <sofa/component/topology/mapping/SubsetTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int SubsetTopologicalMappingClass = core::RegisterObject("This class is a specific implementation of TopologicalMapping where the destination topology is a subset of the source topology. The implementation currently assumes that both topologies have been initialized correctly.")
        .add< SubsetTopologicalMapping >()

        ;

SubsetTopologicalMapping::SubsetTopologicalMapping()
    : d_samePoints(initData(&d_samePoints, false, "samePoints", "True if the same set of points is used in both topologies"))
    , d_handleEdges(initData(&d_handleEdges, false, "handleEdges", "True if edges events and mapping should be handled"))
    , d_handleTriangles(initData(&d_handleTriangles, false, "handleTriangles", "True if triangles events and mapping should be handled"))
    , d_handleQuads(initData(&d_handleQuads, false, "handleQuads", "True if quads events and mapping should be handled"))
    , d_handleTetrahedra(initData(&d_handleTetrahedra, false, "handleTetrahedra", "True if tetrahedra events and mapping should be handled"))
    , d_handleHexahedra(initData(&d_handleHexahedra, false, "handleHexahedra", "True if hexahedra events and mapping should be handled"))
    , d_pointS2D(initData(&d_pointS2D, "pointS2D", "Internal source -> destination topology points map"))
    , d_pointD2S(initData(&d_pointD2S, "pointD2S", "Internal destination -> source topology points map (link to SubsetMapping::indices to handle the mechanical-side of the mapping"))
    , d_edgeS2D(initData(&d_edgeS2D, "edgeS2D", "Internal source -> destination topology edges map"))
    , d_edgeD2S(initData(&d_edgeD2S, "edgeD2S", "Internal destination -> source topology edges map"))
    , d_triangleS2D(initData(&d_triangleS2D, "triangleS2D", "Internal source -> destination topology triangles map"))
    , d_triangleD2S(initData(&d_triangleD2S, "triangleD2S", "Internal destination -> source topology triangles map"))
    , d_quadS2D(initData(&d_quadS2D, "quadS2D", "Internal source -> destination topology quads map"))
    , d_quadD2S(initData(&d_quadD2S, "quadD2S", "Internal destination -> source topology quads map"))
    , d_tetrahedronS2D(initData(&d_tetrahedronS2D, "tetrahedronS2D", "Internal source -> destination topology tetrahedra map"))
    , d_tetrahedronD2S(initData(&d_tetrahedronD2S, "tetrahedronD2S", "Internal destination -> source topology tetrahedra map"))
    , d_hexahedronS2D(initData(&d_hexahedronS2D, "hexahedronS2D", "Internal source -> destination topology hexahedra map"))
    , d_hexahedronD2S(initData(&d_hexahedronD2S, "hexahedronD2S", "Internal destination -> source topology hexahedra map"))
{
    samePoints.setParent(&d_samePoints);
    handleEdges.setParent(&d_handleEdges);
    handleTriangles.setParent(&d_handleTriangles);
    handleQuads.setParent(&d_handleQuads);
    handleTetrahedra.setParent(&d_handleTetrahedra);
    handleHexahedra.setParent(&d_handleHexahedra);
    pointS2D.setParent(&d_pointS2D);
    pointD2S.setParent(&d_pointD2S);
    edgeS2D.setParent(&d_edgeS2D);
    edgeD2S.setParent(&d_edgeD2S);
    triangleS2D.setParent(&d_triangleS2D);
    triangleD2S.setParent(&d_triangleD2S);
    quadS2D.setParent(&d_quadS2D);
    quadD2S.setParent(&d_quadD2S);
    tetrahedronS2D.setParent(&d_tetrahedronS2D);
    tetrahedronD2S.setParent(&d_tetrahedronD2S);
    hexahedronS2D.setParent(&d_hexahedronS2D);
    hexahedronD2S.setParent(&d_hexahedronD2S);
}


SubsetTopologicalMapping::~SubsetTopologicalMapping()
{
}

template<class T> T make_unique(const T& v)
{
    T res = v;
    for (unsigned int i=1; i<res.size(); ++i)
    {
        typename T::value_type r = res[i];
        unsigned int j=i;
        while (j>0 && res[j-1] > r)
        {
            res[j] = res[j-1];
            --j;
        }
        res[j] = r;
    }
    return res;
}

template<class T, class M> T apply_map(const T& v, const M& m)
{
    T res = v;
    if (!m.empty())
        for (unsigned int i=0; i<res.size(); ++i)
            res[i] = m[res[i]];
    return res;
}

void SubsetTopologicalMapping::init()
{
    sofa::core::topology::TopologicalMapping::init();
    this->updateLinks();
    if (fromModel && toModel)
    {
        const Index npS = (Index)fromModel->getNbPoints();
        const Index npD = (Index)  toModel->getNbPoints();
        helper::WriteAccessor<Data<SetIndex> > pS2D(d_pointS2D);
        helper::WriteAccessor<Data<SetIndex> > pD2S(d_pointD2S);
        helper::WriteAccessor<Data<SetIndex> > eS2D(d_edgeS2D);
        helper::WriteAccessor<Data<SetIndex> > eD2S(d_edgeD2S);
        helper::WriteAccessor<Data<SetIndex> > tS2D(d_triangleS2D);
        helper::WriteAccessor<Data<SetIndex> > tD2S(d_triangleD2S);
        helper::WriteAccessor<Data<SetIndex> > qS2D(d_quadS2D);
        helper::WriteAccessor<Data<SetIndex> > qD2S(d_quadD2S);
        helper::WriteAccessor<Data<SetIndex> > teS2D(d_tetrahedronS2D);
        helper::WriteAccessor<Data<SetIndex> > teD2S(d_tetrahedronD2S);
        helper::WriteAccessor<Data<SetIndex> > heS2D(d_hexahedronS2D);
        helper::WriteAccessor<Data<SetIndex> > heD2S(d_hexahedronD2S);
        const Index neS = (Index)fromModel->getNbEdges();
        const Index neD = (Index)  toModel->getNbEdges();
        const Index ntS = (Index)fromModel->getNbTriangles();
        const Index ntD = (Index)  toModel->getNbTriangles();
        const Index nqS = (Index)fromModel->getNbQuads();
        const Index nqD = (Index)  toModel->getNbQuads();
        const Index nteS = (Index)fromModel->getNbTetrahedra();
        const Index nteD = (Index)  toModel->getNbTetrahedra();
        const Index nheS = (Index)fromModel->getNbHexahedra();
        const Index nheD = (Index)  toModel->getNbHexahedra();
        if (!d_samePoints.getValue())
        {
            pS2D.resize(npS); pD2S.resize(npD);
        }
        if (d_handleEdges.getValue())
        {
            eS2D.resize(neS); eD2S.resize(neD);
        }
        if (d_handleTriangles.getValue())
        {
            tS2D.resize(ntS); tD2S.resize(ntD);
        }
        if (d_handleQuads.getValue())
        {
            qS2D.resize(nqS); qD2S.resize(nqD);
        }
        if (d_handleTetrahedra.getValue())
        {
            teS2D.resize(nteS); teD2S.resize(nteD);
        }
        if (d_handleHexahedra.getValue())
        {
            heS2D.resize(nheS); heD2S.resize(nheD);
        }
        if (!d_samePoints.getValue())
        {
            if (fromModel->hasPos() && toModel->hasPos())
            {
                std::map<sofa::type::Vec3d,Index> pmapS;
                for (Index ps = 0; ps < npS; ++ps)
                {
                    type::Vec3d key(fromModel->getPX(ps),fromModel->getPY(ps),fromModel->getPZ(ps));
                    pmapS[key] = ps;
                    pS2D[ps] = sofa::InvalidID;
                }
                for (Index pd = 0; pd < npD; ++pd)
                {
                    type::Vec3d key(toModel->getPX(pd),toModel->getPY(pd),toModel->getPZ(pd));
                    std::map<sofa::type::Vec3d,Index>::const_iterator it = pmapS.find(key);
                    if (it == pmapS.end())
                    {
                        msg_error() << "Point " << pd << " not found in source topology";
                        pD2S[pd] = sofa::InvalidID;
                    }
                    else
                    {
                        Index ps = it->second;
                        pD2S[pd] = ps; pS2D[ps] = pd;
                    }
                }
            }
            else if (npS == npD)
            {
                for (Index pd = 0; pd < npD; ++pd)
                {
                    Index ps = pd;
                    pD2S[pd] = ps; pS2D[ps] = pd;
                }
            }
            else
            {
                msg_error() << "If topologies do not have the same number of points then they must have associated positions";
                return;
            }
        }
        if (d_handleEdges.getValue() && neS > 0 && neD > 0)
        {
            std::map<core::topology::Topology::Edge,Index> emapS;
            for (Index es = 0; es < neS; ++es)
            {
                core::topology::Topology::Edge key(make_unique(fromModel->getEdge(es)));
                emapS[key] = es;
                eS2D[es] = sofa::InvalidID;
            }
            for (Index ed = 0; ed < neD; ++ed)
            {
                core::topology::Topology::Edge key(make_unique(apply_map(toModel->getEdge(ed), pD2S)));
                std::map<core::topology::Topology::Edge,Index>::const_iterator it = emapS.find(key);
                if (it == emapS.end())
                {
                    msg_error() << "Edge " << ed << " not found in source topology";
                    eD2S[ed] = sofa::InvalidID;
                }
                else
                {
                    Index es = it->second;
                    eD2S[ed] = es; eS2D[es] = ed;
                }
            }
        }
        if (d_handleTriangles.getValue() && ntS > 0 && ntD > 0)
        {
            std::map<core::topology::Topology::Triangle,Index> tmapS;
            for (Index ts = 0; ts < ntS; ++ts)
            {
                core::topology::Topology::Triangle key(make_unique(fromModel->getTriangle(ts)));
                tmapS[key] = ts;
                tS2D[ts] = sofa::InvalidID;
            }
            for (Index td = 0; td < ntD; ++td)
            {
                core::topology::Topology::Triangle key(make_unique(apply_map(toModel->getTriangle(td), pD2S)));
                std::map<core::topology::Topology::Triangle,Index>::const_iterator it = tmapS.find(key);
                if (it == tmapS.end())
                {
                    msg_error() << "Triangle " << td << " not found in source topology";
                    tD2S[td] = sofa::InvalidID;
                }
                else
                {
                    Index ts = it->second;
                    tD2S[td] = ts; tS2D[ts] = td;
                }
            }
        }
        if (d_handleQuads.getValue() && nqS > 0 && nqD > 0)
        {
            std::map<core::topology::Topology::Quad,Index> qmapS;
            for (Index qs = 0; qs < nqS; ++qs)
            {
                core::topology::Topology::Quad key(make_unique(fromModel->getQuad(qs)));
                qmapS[key] = qs;
                qS2D[qs] = sofa::InvalidID;
            }
            for (Index qd = 0; qd < nqD; ++qd)
            {
                core::topology::Topology::Quad key(make_unique(apply_map(toModel->getQuad(qd), pD2S)));
                std::map<core::topology::Topology::Quad,Index>::const_iterator it = qmapS.find(key);
                if (it == qmapS.end())
                {
                    msg_error() << "Quad " << qd << " not found in source topology";
                    qD2S[qd] = sofa::InvalidID;
                }
                else
                {
                    Index qs = it->second;
                    qD2S[qd] = qs; qS2D[qs] = qd;
                }
            }
        }
        if (d_handleTetrahedra.getValue() && nteS > 0 && nteD > 0)
        {
            std::map<core::topology::Topology::Tetrahedron,Index> temapS;
            for (Index tes = 0; tes < nteS; ++tes)
            {
                core::topology::Topology::Tetrahedron key(make_unique(fromModel->getTetrahedron(tes)));
                temapS[key] = tes;
                teS2D[tes] = sofa::InvalidID;
            }
            for (Index ted = 0; ted < nteD; ++ted)
            {
                core::topology::Topology::Tetrahedron key(make_unique(apply_map(toModel->getTetrahedron(ted), pD2S)));
                std::map<core::topology::Topology::Tetrahedron,Index>::const_iterator it = temapS.find(key);
                if (it == temapS.end())
                {
                    msg_error() << "Tetrahedron " << ted << " not found in source topology";
                    teD2S[ted] = sofa::InvalidID;
                }
                else
                {
                    Index tes = it->second;
                    teD2S[ted] = tes; teS2D[tes] = ted;
                }
            }
        }
        if (d_handleHexahedra.getValue() && nheS > 0 && nheD > 0)
        {
            std::map<core::topology::Topology::Hexahedron,Index> hemapS;
            for (Index hes = 0; hes < nheS; ++hes)
            {
                core::topology::Topology::Hexahedron key(make_unique(fromModel->getHexahedron(hes)));
                hemapS[key] = hes;
                heS2D[hes] = sofa::InvalidID;
            }
            for (Index hed = 0; hed < nheD; ++hed)
            {
                core::topology::Topology::Hexahedron key(make_unique(apply_map(toModel->getHexahedron(hed), pD2S)));
                std::map<core::topology::Topology::Hexahedron,Index>::const_iterator it = hemapS.find(key);
                if (it == hemapS.end())
                {
                    msg_error() << "Hexahedron " << hed << " not found in source topology";
                    heD2S[hed] = sofa::InvalidID;
                }
                else
                {
                    Index hes = it->second;
                    heD2S[hed] = hes; heS2D[hes] = hed;
                }
            }
        }
        if (!d_samePoints.getValue())
            msg_info() << " P: "<<fromModel->getNbPoints() << "->" << toModel->getNbPoints() << "/" << (fromModel->getNbPoints() - toModel->getNbPoints());
        if (d_handleEdges.getValue())
            msg_info() << " E: "<<fromModel->getNbEdges() << "->" << toModel->getNbEdges() << "/" << (fromModel->getNbEdges() - toModel->getNbEdges());
        if (d_handleTriangles.getValue())
            msg_info() << " T: "<<fromModel->getNbTriangles() << "->" << toModel->getNbTriangles() << "/" << (fromModel->getNbTriangles() - toModel->getNbTriangles());
        if (d_handleQuads.getValue())
            msg_info() << " Q: "<<fromModel->getNbQuads() << "->" << toModel->getNbQuads() << "/" << (fromModel->getNbQuads() - toModel->getNbQuads());
        if (d_handleTetrahedra.getValue())
            msg_info() << " TE: "<<fromModel->getNbTetrahedra() << "->" << toModel->getNbTetrahedra() << "/" << (fromModel->getNbTetrahedra() - toModel->getNbTetrahedra());
        if (d_handleHexahedra.getValue())
            msg_info() << " HE: "<<fromModel->getNbHexahedra() << "->" << toModel->getNbHexahedra() << "/" << (fromModel->getNbHexahedra() - toModel->getNbHexahedra());

        // Need to fully init the target topology
        toModel->init();
    }
}

Index SubsetTopologicalMapping::getFromIndex(Index ind)
{
    return ind;
}

Index SubsetTopologicalMapping::getGlobIndex(Index ind)
{
    if (d_handleHexahedra.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > heD2S(d_hexahedronD2S);
        if (!heD2S.empty()) return heD2S[ind];
    }
    if (d_handleTetrahedra.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > teD2S(d_tetrahedronD2S);
        if (!teD2S.empty()) return teD2S[ind];
    }
    if (d_handleQuads.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > qD2S(d_quadD2S);
        if (!qD2S.empty()) return qD2S[ind];
    }
    if (d_handleTriangles.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > tD2S(d_triangleD2S);
        if (!tD2S.empty()) return tD2S[ind];
    }
    if (d_handleEdges.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > eD2S(d_edgeD2S);
        if (!eD2S.empty()) return eD2S[ind];
    }
    if (!d_samePoints.getValue())
    {
        const helper::ReadAccessor<Data<SetIndex> > pD2S(d_pointD2S);
        if (!pD2S.empty()) return pD2S[ind];
    }
    return ind;
}

void SubsetTopologicalMapping::updateTopologicalMappingTopDown()
{
    using namespace container::dynamic;

    if (!fromModel || !toModel) return;

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    if (itBegin == itEnd) return;

    PointSetTopologyModifier *toPointMod = nullptr;
    EdgeSetTopologyModifier *toEdgeMod = nullptr;
    TriangleSetTopologyModifier *toTriangleMod = nullptr;
    //QuadSetTopologyModifier *toQuadMod = nullptr;
    //TetrahedronSetTopologyModifier *toTetrahedronMod = nullptr;
    //HexahedronSetTopologyModifier *toHexahedronMod = nullptr;

    toModel->getContext()->get(toPointMod);
    if (!toPointMod)
    {
        msg_error() << "No PointSetTopologyModifier found for target topology.";
        return;
    }

    helper::WriteAccessor<Data<SetIndex> > pS2D(d_pointS2D);
    helper::WriteAccessor<Data<SetIndex> > pD2S(d_pointD2S);
    helper::WriteAccessor<Data<SetIndex> > eS2D(d_edgeS2D);
    helper::WriteAccessor<Data<SetIndex> > eD2S(d_edgeD2S);
    helper::WriteAccessor<Data<SetIndex> > tS2D(d_triangleS2D);
    helper::WriteAccessor<Data<SetIndex> > tD2S(d_triangleD2S);
    helper::WriteAccessor<Data<SetIndex> > qS2D(d_quadS2D);
    helper::WriteAccessor<Data<SetIndex> > qD2S(d_quadD2S);
    helper::WriteAccessor<Data<SetIndex> > teS2D(d_tetrahedronS2D);
    helper::WriteAccessor<Data<SetIndex> > teD2S(d_tetrahedronD2S);
    helper::WriteAccessor<Data<SetIndex> > heS2D(d_hexahedronS2D);
    helper::WriteAccessor<Data<SetIndex> > heD2S(d_hexahedronD2S);
    int count = 0;
    while( itBegin != itEnd )
    {
        const TopologyChange* topoChange = *itBegin;
        TopologyChangeType changeType = topoChange->getChangeType();

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            msg_info() << "[" << count << "]ENDING_EVENT";
            toPointMod->notifyEndingEvent();
            if (!d_samePoints.getValue())
            {
                if (pS2D.size() != (unsigned)fromModel->getNbPoints()) msg_error() << "Invalid d_pointS2D size : " << pS2D.size() << " != " << fromModel->getNbPoints();
                if (pD2S.size() != (unsigned)  toModel->getNbPoints()) msg_error() << "Invalid d_pointD2S size : " << pD2S.size() << " != " <<   toModel->getNbPoints();
            }
            if (d_handleEdges.getValue())
            {
                if (eS2D.size() != (unsigned)fromModel->getNbEdges()) msg_error() << "Invalid d_edgeS2D size : " << eS2D.size() << " != " << fromModel->getNbEdges();
                if (eD2S.size() != (unsigned)  toModel->getNbEdges()) msg_error() << "Invalid d_edgeD2S size : " << eD2S.size() << " != " <<   toModel->getNbEdges();
            }
            if (d_handleTriangles.getValue())
            {
                if (tS2D.size() != (unsigned)fromModel->getNbTriangles()) msg_error() << "Invalid d_triangleS2D size : " << tS2D.size() << " != " << fromModel->getNbTriangles();
                if (tD2S.size() != (unsigned)  toModel->getNbTriangles()) msg_error() << "Invalid d_triangleD2S size : " << tD2S.size() << " != " <<   toModel->getNbTriangles();
            }
            if (d_handleQuads.getValue())
            {
                if (qS2D.size() != (unsigned)fromModel->getNbQuads()) msg_error() << "Invalid d_quadS2D size : " << qS2D.size() << " != " << fromModel->getNbQuads();
                if (qD2S.size() != (unsigned)  toModel->getNbQuads()) msg_error() << "Invalid d_quadD2S size : " << qD2S.size() << " != " <<   toModel->getNbQuads();
            }
            if (d_handleTetrahedra.getValue())
            {
                if (teS2D.size() != (unsigned)fromModel->getNbTetrahedra()) msg_error() << "Invalid d_tetrahedronS2D size : " << teS2D.size() << " != " << fromModel->getNbTetrahedra();
                if (teD2S.size() != (unsigned)  toModel->getNbTetrahedra()) msg_error() << "Invalid d_tetrahedronD2S size : " << teD2S.size() << " != " <<   toModel->getNbTetrahedra();
            }
            if (d_handleHexahedra.getValue())
            {
                if (heS2D.size() != (unsigned)fromModel->getNbHexahedra()) msg_error() << "Invalid d_hexahedronS2D size : " << heS2D.size() << " != " << fromModel->getNbHexahedra();
                if (heD2S.size() != (unsigned)  toModel->getNbHexahedra()) msg_error() << "Invalid d_hexahedronD2S size : " << heD2S.size() << " != " <<   toModel->getNbHexahedra();
            }
            if (!d_samePoints.getValue())
                msg_info() << " P: "<<fromModel->getNbPoints() << "->" << toModel->getNbPoints() << "/" << (fromModel->getNbPoints() - toModel->getNbPoints());
            if (d_handleEdges.getValue())
                msg_info() << " E: "<<fromModel->getNbEdges() << "->" << toModel->getNbEdges() << "/" << (fromModel->getNbEdges() - toModel->getNbEdges());
            if (d_handleTriangles.getValue())
                msg_info() << " T: "<<fromModel->getNbTriangles() << "->" << toModel->getNbTriangles() << "/" << (fromModel->getNbTriangles() - toModel->getNbTriangles());
            if (d_handleQuads.getValue())
                msg_info() << " Q: "<<fromModel->getNbQuads() << "->" << toModel->getNbQuads() << "/" << (fromModel->getNbQuads() - toModel->getNbQuads());
            if (d_handleTetrahedra.getValue())
                msg_info() << " TE: "<<fromModel->getNbTetrahedra() << "->" << toModel->getNbTetrahedra() << "/" << (fromModel->getNbTetrahedra() - toModel->getNbTetrahedra());
            if (d_handleHexahedra.getValue())
                msg_info() << " HE: "<<fromModel->getNbHexahedra() << "->" << toModel->getNbHexahedra() << "/" << (fromModel->getNbHexahedra() - toModel->getNbHexahedra());
            break;
        }

        case core::topology::POINTSADDED:
        {
            const PointsAdded * pAdd = static_cast< const PointsAdded * >( topoChange );
            size_t pS0 = (d_samePoints.getValue() ? toModel->getNbPoints() : pS2D.size());
            size_t nSadd = pAdd->getNbAddedVertices();
            msg_info() << "[" << count << "]POINTSADDED : " << nSadd << " : " << pS0 << " - " << (pS0 + nSadd-1);
            if (d_samePoints.getValue())
            {
                toPointMod->addPoints(pAdd->getNbAddedVertices());
            }
            else
            {
                type::vector< type::vector<Index> > ancestors;
                type::vector< type::vector<SReal> > coefs;
                size_t nDadd = 0;
                size_t pD0 = pD2S.size();
                pS2D.resize(pS0+nSadd);
                ancestors.reserve(pAdd->ancestorsList.size());
                coefs.reserve(pAdd->coefs.size());
                for (Index pi = 0; pi < nSadd; ++pi)
                {
                    size_t ps = pS0+pi;
                    pS2D[ps] = sofa::InvalidID;
                    bool inDst = true;
                    if (!pAdd->ancestorsList.empty())
                        for (unsigned int i=0; i<pAdd->ancestorsList[pi].size() && inDst; ++i)
                        {
                            if (pAdd->coefs[pi][i] == 0.0) continue;
                            if (pS2D[pAdd->ancestorsList[pi][i]] == sofa::InvalidID)
                                inDst = false;
                        }
                    if (!inDst) continue;
                    size_t pd = pD0+nDadd;
                    pD2S.resize(pd+1);
                    pS2D[ps] = (unsigned int)pd;
                    pD2S[pd] = (unsigned int)ps;

                    if (!pAdd->ancestorsList.empty())
                    {
                        ancestors.resize(nDadd+1);
                        coefs.resize(nDadd+1);
                        for (unsigned int i=0; i<pAdd->ancestorsList[pi].size(); ++i)
                        {
                            if (pAdd->coefs[pi][i] == 0.0) continue;
                            ancestors[nDadd].push_back(pS2D[pAdd->ancestorsList[pi][i]]);
                            coefs[nDadd].push_back(pAdd->coefs[pi][i]);
                        }
                    }
                    ++nDadd;
                }
                if (nDadd > 0)
                {
                    msg_info() << "    -> POINTSADDED : " << nDadd << " : " << pD0 << " - " << (pD0 + nDadd-1);
                    toPointMod->addPoints(nDadd, ancestors, coefs, true);
                }
            }
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const PointsRemoved *pRem = static_cast< const PointsRemoved * >( topoChange );
            sofa::type::vector<Index> tab = pRem->getArray();
            if (d_samePoints.getValue())
            {
                msg_info() << "[" << count << "]POINTSREMOVED : " << tab.size() << " : " << tab;
                toPointMod->removePoints(tab, true);
            }
            else
            {
                sofa::type::vector<Index> tab2;
                tab2.reserve(tab.size());
                for (unsigned int pi=0; pi<tab.size(); ++pi)
                {
                    Index ps = tab[pi];
                    Index pd = pS2D[ps];
                    if (pd == sofa::InvalidID)
                        continue;
                    tab2.push_back(pd);
                }
                msg_info() << "[" << count << "]POINTSREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2;
                // apply removals in pS2D
                {
                    size_t last = pS2D.size() -1;
                    for (unsigned int i = 0; i < tab.size(); ++i)
                    {
                        Index ps = tab[i];
                        Index pd = pS2D[ps];
                        if (pd != sofa::InvalidID)
                            pD2S[pd] = sofa::InvalidID;
                        Index pd2 = pS2D[last];
                        pS2D[ps] = pd2;
                        if (pd2 != sofa::InvalidID && pD2S[pd2] == last)
                            pD2S[pd2] = ps;
                        --last;
                    }
                    pS2D.resize( pS2D.size() - tab.size() );
                }
                if (!tab2.empty())
                {
                    toPointMod->removePoints(tab2, true);
                    // apply removals in pD2S
                    {
                        size_t last = pD2S.size() -1;
                        for (unsigned int i = 0; i < tab2.size(); ++i)
                        {
                            Index pd = tab2[i];
                            Index ps = pD2S[pd];
                            if (ps != sofa::InvalidID)
                                msg_error() << "Invalid Point Remove";
                            Index ps2 = pD2S[last];
                            pD2S[pd] = ps2;
                            if (ps2 != sofa::InvalidID && pS2D[ps2] == last)
                                pS2D[ps2] = pd;
                            --last;
                        }
                        pD2S.resize( pD2S.size() - tab2.size() );
                    }
                }
            }
            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const PointsRenumbering *pRenumber = static_cast< const PointsRenumbering * >( topoChange );
            const sofa::type::vector<Index> &tab = pRenumber->getIndexArray();
            const sofa::type::vector<Index> &inv_tab = pRenumber->getinv_IndexArray();
            if (d_samePoints.getValue())
            {
                msg_info() << "[" << count << "]POINTSRENUMBERING : " << tab.size() << " : " << tab;
                toPointMod->renumberPoints(tab, inv_tab, true);
            }
            else
            {
                sofa::type::vector<Index> tab2;
                sofa::type::vector<Index> inv_tab2;
                tab2.resize(pD2S.size());
                inv_tab2.resize(pD2S.size());
                for (Index pd = 0; pd < pD2S.size(); ++pd)
                {
                    Index ps = pD2S[pd];
                    Index ps2 = (ps == sofa::InvalidID) ? sofa::InvalidID : tab[ps];
                    Index pd2 = (ps2 == sofa::InvalidID) ? sofa::InvalidID : pS2D[ps2];
                    if (pd2 == sofa::InvalidID) pd2 = pd;
                    tab2[pd] = pd2;
                    inv_tab2[pd2] = pd;
                }
                msg_info() << "[" << count << "]POINTSRENUMBERING : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2;
                toPointMod->renumberPoints(tab2, inv_tab2, true);
                SetIndex pS2D0 = pS2D.ref();
                for (Index ps = 0; ps < pS2D.size(); ++ps)
                {
                    Index pd = pS2D0[tab2[ps]];
                    if (pd != sofa::InvalidID)
                        pd = inv_tab2[pd];
                    pS2D[ps] = pd;
                }
                SetIndex pD2S0 = pD2S.ref();
                for (Index pd = 0; pd < pD2S.size(); ++pd)
                {
                    Index ps = pD2S0[tab[pd]];
                    if (ps != sofa::InvalidID)
                        ps = inv_tab[ps];
                    pD2S[pd] = ps;
                }
            }
            break;
        }

        case core::topology::EDGESADDED:
        {
            if (!d_handleEdges.getValue()) break;
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesAdded *eAdd = static_cast< const EdgesAdded * >( topoChange );
            msg_info() << "[" << count << "]EDGESADDED : " << eAdd->getNbAddedEdges() << " : " << eAdd->edgeIndexArray << " : " << eAdd->edgeArray;
            //toEdgeMod->addEdgesProcess(eAdd->edgeArray);
            //toEdgeMod->addEdgesWarning(eAdd->getNbAddedEdges(), eAdd->edgeArray, eAdd->edgeIndexArray, eAdd->ancestorsList, eAdd->coefs);
            //toEdgeMod->propagateTopologicalChanges();
            type::vector< core::topology::BaseMeshTopology::Edge > edgeArray;
            type::vector< Index > edgeIndexArray;
            type::vector< type::vector<Index> > ancestors;
            type::vector< type::vector<SReal> > coefs;
            size_t nSadd = eAdd->getNbAddedEdges();
            unsigned int nDadd = 0;
            size_t eS0 = eS2D.size();
            size_t eD0 = eD2S.size();
            eS2D.resize(eS0+nSadd);
            edgeArray.reserve(eAdd->edgeArray.size());
            edgeIndexArray.reserve(eAdd->edgeIndexArray.size());
            ancestors.reserve(eAdd->ancestorsList.size());
            coefs.reserve(eAdd->coefs.size());
            for (Index ei = 0; ei < nSadd; ++ei)
            {
                Index es = eAdd->edgeIndexArray[ei];
                eS2D[es] = sofa::InvalidID;
                bool inDst = true;
                core::topology::BaseMeshTopology::Edge data = apply_map(eAdd->edgeArray[ei], pS2D);
                for (unsigned int i=0; i<data.size() && inDst; ++i)
                    if (data[i] == sofa::InvalidID)
                        inDst = false;
                if (!eAdd->ancestorsList.empty())
                    for (unsigned int i=0; i<eAdd->ancestorsList[ei].size() && inDst; ++i)
                    {
                        if (eAdd->coefs[ei][i] == 0.0) continue;
                        if (eS2D[eAdd->ancestorsList[ei][i]] == sofa::InvalidID)
                            inDst = false;
                    }
                if (!inDst) continue;
                unsigned int ed = (unsigned int)(eD0+nDadd);
                edgeArray.push_back(data);
                edgeIndexArray.push_back(ed);
                eD2S.resize(ed+1);
                eS2D[es] = ed;
                eD2S[ed] = es;
                if (!eAdd->ancestorsList.empty())
                {
                    ancestors.resize(nDadd+1);
                    coefs.resize(nDadd+1);
                    for (unsigned int i=0; i<eAdd->ancestorsList[ei].size(); ++i)
                    {
                        if (eAdd->coefs[ei][i] == 0.0) continue;
                        ancestors[nDadd].push_back(eS2D[eAdd->ancestorsList[ei][i]]);
                        coefs[nDadd].push_back(eAdd->coefs[ei][i]);
                    }
                }
                ++nDadd;
            }
            if (nDadd > 0)
            {
                msg_info() << "    -> EDGESADDED : " << nDadd << " : " << edgeIndexArray << " : " << edgeArray;
                toEdgeMod->addEdges(edgeArray, ancestors, coefs);
            }
            break;
        }

        case core::topology::EDGESREMOVED:
        {
            if (!d_handleEdges.getValue()) break;
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesRemoved *eRem = static_cast< const EdgesRemoved * >( topoChange );
            sofa::type::vector<Index> tab = eRem->getArray();
            //toEdgeMod->removeEdgesWarning(tab);
            //toEdgeMod->propagateTopologicalChanges();
            //toEdgeMod->removeEdgesProcess(tab, false);
            sofa::type::vector<Index> tab2;
            tab2.reserve(tab.size());
            for (unsigned int ei=0; ei<tab.size(); ++ei)
            {
                Index es = tab[ei];
                Index ed = eS2D[es];
                if (ed == sofa::InvalidID)
                    continue;
                tab2.push_back(ed);
            }
            msg_info() << "[" << count << "]EDGESREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2;
            // apply removals in eS2D
            {
                size_t last = eS2D.size() -1;
                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    Index es = tab[i];
                    Index ed = eS2D[es];
                    if (ed != sofa::InvalidID)
                        eD2S[ed] = sofa::InvalidID;
                    Index ed2 = eS2D[last];
                    eS2D[es] = ed2;
                    if (ed2 != sofa::InvalidID && eD2S[ed2] == last)
                        eD2S[ed2] = es;
                    --last;
                }
                eS2D.resize( eS2D.size() - tab.size() );
            }
            if (!tab2.empty())
            {
                toEdgeMod->removeEdges(tab2, false);
                // apply removals in eD2S
                {
                    size_t last = eD2S.size() -1;
                    for (unsigned int i = 0; i < tab2.size(); ++i)
                    {
                        Index ed = tab2[i];
                        Index es = eD2S[ed];
                        if (es != sofa::InvalidID)
                            msg_error() << "Invalid Edge Remove";
                        Index es2 = eD2S[last];
                        eD2S[ed] = es2;
                        if (es2 != sofa::InvalidID && eS2D[es2] == last)
                            eS2D[es2] = ed;
                        --last;
                    }
                    eD2S.resize( eD2S.size() - tab2.size() );
                }
            }

            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            if (!d_handleTriangles.getValue()) break;
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesAdded *tAdd = static_cast< const TrianglesAdded * >( topoChange );
            msg_info() << "[" << count << "]TRIANGLESADDED : " << tAdd->getNbAddedTriangles() << " : " << tAdd->triangleIndexArray << " : " << tAdd->triangleArray;
            if (!tAdd->ancestorsList.empty())
            {
                size_t count = 0;
                double sum = 0.0;
                msg_info() << "   ";
                for (unsigned int i = 0; i < tAdd->ancestorsList.size(); ++i)
                {
                    msg_info() << "    " << tAdd->ancestorsList[i];
                    count += tAdd->ancestorsList[i].size();
                    for (unsigned int j=0; j<tAdd->ancestorsList[i].size(); ++j)
                        sum += tAdd->coefs[i][j];
                }
                msg_info() << "     " << tAdd->ancestorsList.size() << " ancestor lists specified, " << count << " ancestors total, " << sum/tAdd->ancestorsList.size() << " avg coefs sum";
            }
            //toTriangleMod->addTrianglesProcess(tAdd->triangleArray);
            //toTriangleMod->addTrianglesWarning(tAdd->getNbAddedTriangles(), tAdd->triangleArray, tAdd->triangleIndexArray, tAdd->ancestorsList, tAdd->coefs);
            //toTriangleMod->propagateTopologicalChanges();
            type::vector< core::topology::BaseMeshTopology::Triangle > triangleArray;
            type::vector< Index > triangleIndexArray;
            type::vector< type::vector<Index> > ancestors;
            type::vector< type::vector<SReal> > coefs;
            size_t nSadd = tAdd->getNbAddedTriangles();
            unsigned int nDadd = 0;
            size_t tS0 = tS2D.size();
            size_t tD0 = tD2S.size();
            tS2D.resize(tS0+nSadd);
            triangleArray.reserve(tAdd->triangleArray.size());
            triangleIndexArray.reserve(tAdd->triangleIndexArray.size());
            ancestors.reserve(tAdd->ancestorsList.size());
            coefs.reserve(tAdd->coefs.size());
            for (Index ti = 0; ti < nSadd; ++ti)
            {
                Index ts = tAdd->triangleIndexArray[ti];
                tS2D[ts] = sofa::InvalidID;
                bool inDst = true;
                core::topology::BaseMeshTopology::Triangle data = apply_map(tAdd->triangleArray[ti], pS2D);
                for (unsigned int i=0; i<data.size() && inDst; ++i)
                    if (data[i] == sofa::InvalidID)
                        inDst = false;
                if (!tAdd->ancestorsList.empty())
                    for (unsigned int i=0; i<tAdd->ancestorsList[ti].size() && inDst; ++i)
                    {
                        if (tAdd->coefs[ti][i] == 0.0) continue;
                        if (tS2D[tAdd->ancestorsList[ti][i]] == sofa::InvalidID)
                            inDst = false;
                    }
                if (!inDst) continue;
                unsigned int td = (unsigned int)(tD0+nDadd);
                triangleArray.push_back(data);
                triangleIndexArray.push_back(td);
                tD2S.resize(td+1);
                tS2D[ts] = td;
                tD2S[td] = ts;
                if (!tAdd->ancestorsList.empty())
                {
                    ancestors.resize(nDadd+1);
                    coefs.resize(nDadd+1);
                    for (unsigned int i=0; i<tAdd->ancestorsList[ti].size(); ++i)
                    {
                        if (tAdd->coefs[ti][i] == 0.0) continue;
                        ancestors[nDadd].push_back(tS2D[tAdd->ancestorsList[ti][i]]);
                        coefs[nDadd].push_back(tAdd->coefs[ti][i]);
                    }
                }
                ++nDadd;
            }
            if (nDadd > 0)
            {
                msg_info() << "    -> TRIANGLESADDED : " << nDadd  << " : " << triangleIndexArray << " : " << triangleArray;
                toTriangleMod->addTriangles(triangleArray, ancestors, coefs);
            }
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!d_handleTriangles.getValue()) break;
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesRemoved *tRem = static_cast< const TrianglesRemoved * >( topoChange );
            sofa::type::vector<Index> tab = tRem->getArray();
            //toTriangleMod->removeTrianglesWarning(tab);
            //toTriangleMod->propagateTopologicalChanges();
            //toTriangleMod->removeTrianglesProcess(tab, false);
            sofa::type::vector<Index> tab2;
            tab2.reserve(tab.size());
            for (unsigned int ti=0; ti<tab.size(); ++ti)
            {
                Index ts = tab[ti];
                Index td = tS2D[ts];
                if (td == sofa::InvalidID)
                    continue;
                tab2.push_back(td);
            }
            msg_info() << "[" << count << "]TRIANGLESREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2;
            // apply removals in tS2D
            {
                size_t last = tS2D.size() -1;
                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    Index ts = tab[i];
                    Index td = tS2D[ts];
                    if (td != sofa::InvalidID)
                        tD2S[td] = sofa::InvalidID;
                    Index td2 = tS2D[last];
                    tS2D[ts] = td2;
                    if (td2 != sofa::InvalidID && tD2S[td2] == last)
                        tD2S[td2] = ts;
                    --last;
                }
                tS2D.resize( tS2D.size() - tab.size() );
            }
            if (!tab2.empty())
            {
                toTriangleMod->removeTriangles(tab2, !d_handleEdges.getValue(), false);
                // apply removals in tD2S
                {
                    size_t last = tD2S.size() -1;
                    for (unsigned int i = 0; i < tab2.size(); ++i)
                    {
                        Index td = tab2[i];
                        Index ts = tD2S[td];
                        if (ts != sofa::InvalidID)
                            msg_error() << "Invalid Triangle Remove";
                        Index ts2 = tD2S[last];
                        tD2S[td] = ts2;
                        if (ts2 != sofa::InvalidID && tS2D[ts2] == last)
                            tS2D[ts2] = td;
                        --last;
                    }
                    tD2S.resize( tD2S.size() - tab2.size() );
                }
            }
            break;
        }

        default:
            msg_error() << "Unknown topological change " << changeType;
            break;
        };
        ++count;
        ++itBegin;
    }
}

} //namespace sofa::component::topology::mapping
