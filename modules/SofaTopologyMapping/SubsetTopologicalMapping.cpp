/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTopologyMapping/SubsetTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

SOFA_DECL_CLASS(SubsetTopologicalMapping)

// Register in the Factory
int SubsetTopologicalMappingClass = core::RegisterObject("This class is a specific implementation of TopologicalMapping where the destination topology is a subset of the source topology. The implementation currently assumes that both topologies have been initialized correctly.")
        .add< SubsetTopologicalMapping >()

        ;

SubsetTopologicalMapping::SubsetTopologicalMapping()
    : samePoints(initData(&samePoints,false,"samePoints", "True if the same set of points is used in both topologies"))
    , handleEdges(initData(&handleEdges,false,"handleEdges", "True if edges events and mapping should be handled"))
    , handleTriangles(initData(&handleTriangles,false,"handleTriangles", "True if triangles events and mapping should be handled"))
    , handleQuads(initData(&handleQuads,false,"handleQuads", "True if quads events and mapping should be handled"))
    , handleTetrahedra(initData(&handleTetrahedra,false,"handleTetrahedra", "True if tetrahedra events and mapping should be handled"))
    , handleHexahedra(initData(&handleHexahedra,false,"handleHexahedra", "True if hexahedra events and mapping should be handled"))
    , pointS2D(initData(&pointS2D,"pointS2D", "Internal source -> destination topology points map"))
    , pointD2S(initData(&pointD2S,"pointD2S", "Internal destination -> source topology points map (link to SubsetMapping::indices to handle the mechanical-side of the mapping"))
    , edgeS2D(initData(&edgeS2D,"edgeS2D", "Internal source -> destination topology edges map"))
    , edgeD2S(initData(&edgeD2S,"edgeD2S", "Internal destination -> source topology edges map"))
    , triangleS2D(initData(&triangleS2D,"triangleS2D", "Internal source -> destination topology triangles map"))
    , triangleD2S(initData(&triangleD2S,"triangleD2S", "Internal destination -> source topology triangles map"))
    , quadS2D(initData(&quadS2D,"quadS2D", "Internal source -> destination topology quads map"))
    , quadD2S(initData(&quadD2S,"quadD2S", "Internal destination -> source topology quads map"))
    , tetrahedronS2D(initData(&tetrahedronS2D,"tetrahedronS2D", "Internal source -> destination topology tetrahedra map"))
    , tetrahedronD2S(initData(&tetrahedronD2S,"tetrahedronD2S", "Internal destination -> source topology tetrahedra map"))
    , hexahedronS2D(initData(&hexahedronS2D,"hexahedronS2D", "Internal source -> destination topology hexahedra map"))
    , hexahedronD2S(initData(&hexahedronD2S,"hexahedronD2S", "Internal destination -> source topology hexahedra map"))
{
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
        helper::WriteAccessor<Data<SetIndex> > pS2D(pointS2D);
        helper::WriteAccessor<Data<SetIndex> > pD2S(pointD2S);
        helper::WriteAccessor<Data<SetIndex> > eS2D(edgeS2D);
        helper::WriteAccessor<Data<SetIndex> > eD2S(edgeD2S);
        helper::WriteAccessor<Data<SetIndex> > tS2D(triangleS2D);
        helper::WriteAccessor<Data<SetIndex> > tD2S(triangleD2S);
        helper::WriteAccessor<Data<SetIndex> > qS2D(quadS2D);
        helper::WriteAccessor<Data<SetIndex> > qD2S(quadD2S);
        helper::WriteAccessor<Data<SetIndex> > teS2D(tetrahedronS2D);
        helper::WriteAccessor<Data<SetIndex> > teD2S(tetrahedronD2S);
        helper::WriteAccessor<Data<SetIndex> > heS2D(hexahedronS2D);
        helper::WriteAccessor<Data<SetIndex> > heD2S(hexahedronD2S);
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
        if (!samePoints.getValue())
        {
            pS2D.resize(npS); pD2S.resize(npD);
        }
        if (handleEdges.getValue())
        {
            eS2D.resize(neS); eD2S.resize(neD);
        }
        if (handleTriangles.getValue())
        {
            tS2D.resize(ntS); tD2S.resize(ntD);
        }
        if (handleQuads.getValue())
        {
            qS2D.resize(nqS); qD2S.resize(nqD);
        }
        if (handleTetrahedra.getValue())
        {
            teS2D.resize(nteS); teD2S.resize(nteD);
        }
        if (handleHexahedra.getValue())
        {
            heS2D.resize(nheS); heD2S.resize(nheD);
        }
        if (!samePoints.getValue())
        {
            if (fromModel->hasPos() && toModel->hasPos())
            {
                std::map<sofa::defaulttype::Vec3d,Index> pmapS;
                for (Index ps = 0; ps < npS; ++ps)
                {
                    defaulttype::Vec3d key(fromModel->getPX(ps),fromModel->getPY(ps),fromModel->getPZ(ps));
                    pmapS[key] = ps;
                    pS2D[ps] = core::topology::Topology::InvalidID;
                }
                for (Index pd = 0; pd < npD; ++pd)
                {
                    defaulttype::Vec3d key(toModel->getPX(pd),toModel->getPY(pd),toModel->getPZ(pd));
                    std::map<sofa::defaulttype::Vec3d,Index>::const_iterator it = pmapS.find(key);
                    if (it == pmapS.end())
                    {
                        serr << "Point " << pd << " not found in source topology" << sendl;
                        pD2S[pd] = core::topology::Topology::InvalidID;
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
                serr << "If topologies do not have the same number of points then they must have associated positions" << sendl;
                return;
            }
        }
        if (handleEdges.getValue() && neS > 0 && neD > 0)
        {
            std::map<core::topology::Topology::Edge,Index> emapS;
            for (Index es = 0; es < neS; ++es)
            {
                core::topology::Topology::Edge key(make_unique(fromModel->getEdge(es)));
                emapS[key] = es;
                eS2D[es] = core::topology::Topology::InvalidID;
            }
            for (Index ed = 0; ed < neD; ++ed)
            {
                core::topology::Topology::Edge key(make_unique(apply_map(toModel->getEdge(ed), pD2S)));
                std::map<core::topology::Topology::Edge,Index>::const_iterator it = emapS.find(key);
                if (it == emapS.end())
                {
                    serr << "Edge " << ed << " not found in source topology" << sendl;
                    eD2S[ed] = core::topology::Topology::InvalidID;
                }
                else
                {
                    Index es = it->second;
                    eD2S[ed] = es; eS2D[es] = ed;
                }
            }
        }
        if (handleTriangles.getValue() && ntS > 0 && ntD > 0)
        {
            std::map<core::topology::Topology::Triangle,Index> tmapS;
            for (Index ts = 0; ts < ntS; ++ts)
            {
                core::topology::Topology::Triangle key(make_unique(fromModel->getTriangle(ts)));
                tmapS[key] = ts;
                tS2D[ts] = core::topology::Topology::InvalidID;
            }
            for (Index td = 0; td < ntD; ++td)
            {
                core::topology::Topology::Triangle key(make_unique(apply_map(toModel->getTriangle(td), pD2S)));
                std::map<core::topology::Topology::Triangle,Index>::const_iterator it = tmapS.find(key);
                if (it == tmapS.end())
                {
                    serr << "Triangle " << td << " not found in source topology" << sendl;
                    tD2S[td] = core::topology::Topology::InvalidID;
                }
                else
                {
                    Index ts = it->second;
                    tD2S[td] = ts; tS2D[ts] = td;
                }
            }
        }
        if (handleQuads.getValue() && nqS > 0 && nqD > 0)
        {
            std::map<core::topology::Topology::Quad,Index> qmapS;
            for (Index qs = 0; qs < nqS; ++qs)
            {
                core::topology::Topology::Quad key(make_unique(fromModel->getQuad(qs)));
                qmapS[key] = qs;
                qS2D[qs] = core::topology::Topology::InvalidID;
            }
            for (Index qd = 0; qd < nqD; ++qd)
            {
                core::topology::Topology::Quad key(make_unique(apply_map(toModel->getQuad(qd), pD2S)));
                std::map<core::topology::Topology::Quad,Index>::const_iterator it = qmapS.find(key);
                if (it == qmapS.end())
                {
                    serr << "Quad " << qd << " not found in source topology" << sendl;
                    qD2S[qd] = core::topology::Topology::InvalidID;
                }
                else
                {
                    Index qs = it->second;
                    qD2S[qd] = qs; qS2D[qs] = qd;
                }
            }
        }
        if (handleTetrahedra.getValue() && nteS > 0 && nteD > 0)
        {
            std::map<core::topology::Topology::Tetrahedron,Index> temapS;
            for (Index tes = 0; tes < nteS; ++tes)
            {
                core::topology::Topology::Tetrahedron key(make_unique(fromModel->getTetrahedron(tes)));
                temapS[key] = tes;
                teS2D[tes] = core::topology::Topology::InvalidID;
            }
            for (Index ted = 0; ted < nteD; ++ted)
            {
                core::topology::Topology::Tetrahedron key(make_unique(apply_map(toModel->getTetrahedron(ted), pD2S)));
                std::map<core::topology::Topology::Tetrahedron,Index>::const_iterator it = temapS.find(key);
                if (it == temapS.end())
                {
                    serr << "Tetrahedron " << ted << " not found in source topology" << sendl;
                    teD2S[ted] = core::topology::Topology::InvalidID;
                }
                else
                {
                    Index tes = it->second;
                    teD2S[ted] = tes; teS2D[tes] = ted;
                }
            }
        }
        if (handleHexahedra.getValue() && nheS > 0 && nheD > 0)
        {
            std::map<core::topology::Topology::Hexahedron,Index> hemapS;
            for (Index hes = 0; hes < nheS; ++hes)
            {
                core::topology::Topology::Hexahedron key(make_unique(fromModel->getHexahedron(hes)));
                hemapS[key] = hes;
                heS2D[hes] = core::topology::Topology::InvalidID;
            }
            for (Index hed = 0; hed < nheD; ++hed)
            {
                core::topology::Topology::Hexahedron key(make_unique(apply_map(toModel->getHexahedron(hed), pD2S)));
                std::map<core::topology::Topology::Hexahedron,Index>::const_iterator it = hemapS.find(key);
                if (it == hemapS.end())
                {
                    serr << "Hexahedron " << hed << " not found in source topology" << sendl;
                    heD2S[hed] = core::topology::Topology::InvalidID;
                }
                else
                {
                    Index hes = it->second;
                    heD2S[hed] = hes; heS2D[hes] = hed;
                }
            }
        }
        if (!samePoints.getValue())
            sout << " P: "<<fromModel->getNbPoints() << "->" << toModel->getNbPoints() << "/" << (fromModel->getNbPoints() - toModel->getNbPoints());
        if (handleEdges.getValue())
            sout << " E: "<<fromModel->getNbEdges() << "->" << toModel->getNbEdges() << "/" << (fromModel->getNbEdges() - toModel->getNbEdges());
        if (handleTriangles.getValue())
            sout << " T: "<<fromModel->getNbTriangles() << "->" << toModel->getNbTriangles() << "/" << (fromModel->getNbTriangles() - toModel->getNbTriangles());
        if (handleQuads.getValue())
            sout << " Q: "<<fromModel->getNbQuads() << "->" << toModel->getNbQuads() << "/" << (fromModel->getNbQuads() - toModel->getNbQuads());
        if (handleTetrahedra.getValue())
            sout << " TE: "<<fromModel->getNbTetrahedra() << "->" << toModel->getNbTetrahedra() << "/" << (fromModel->getNbTetrahedra() - toModel->getNbTetrahedra());
        if (handleHexahedra.getValue())
            sout << " HE: "<<fromModel->getNbHexahedra() << "->" << toModel->getNbHexahedra() << "/" << (fromModel->getNbHexahedra() - toModel->getNbHexahedra());
        sout << sendl;
    }
}

unsigned int SubsetTopologicalMapping::getFromIndex(unsigned int ind)
{
    return ind;
}

unsigned int SubsetTopologicalMapping::getGlobIndex(unsigned int ind)
{
    if (handleHexahedra.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > heD2S(hexahedronD2S);
        if (!heD2S.empty()) return heD2S[ind];
    }
    if (handleTetrahedra.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > teD2S(tetrahedronD2S);
        if (!teD2S.empty()) return teD2S[ind];
    }
    if (handleQuads.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > qD2S(quadD2S);
        if (!qD2S.empty()) return qD2S[ind];
    }
    if (handleTriangles.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > tD2S(triangleD2S);
        if (!tD2S.empty()) return tD2S[ind];
    }
    if (handleEdges.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > eD2S(edgeD2S);
        if (!eD2S.empty()) return eD2S[ind];
    }
    if (!samePoints.getValue())
    {
        helper::ReadAccessor<Data<SetIndex> > pD2S(pointD2S);
        if (!pD2S.empty()) return pD2S[ind];
    }
    return ind;
}

void SubsetTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (!fromModel || !toModel) return;

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    if (itBegin == itEnd) return;

    PointSetTopologyModifier *toPointMod = NULL;
    EdgeSetTopologyModifier *toEdgeMod = NULL;
    TriangleSetTopologyModifier *toTriangleMod = NULL;
    //QuadSetTopologyModifier *toQuadMod = NULL;
    //TetrahedronSetTopologyModifier *toTetrahedronMod = NULL;
    //HexahedronSetTopologyModifier *toHexahedronMod = NULL;

    toModel->getContext()->get(toPointMod);
    if (!toPointMod)
    {
        serr << "No PointSetTopologyModifier found for target topology." << sendl;
        return;
    }

    helper::WriteAccessor<Data<SetIndex> > pS2D(pointS2D);
    helper::WriteAccessor<Data<SetIndex> > pD2S(pointD2S);
    helper::WriteAccessor<Data<SetIndex> > eS2D(edgeS2D);
    helper::WriteAccessor<Data<SetIndex> > eD2S(edgeD2S);
    helper::WriteAccessor<Data<SetIndex> > tS2D(triangleS2D);
    helper::WriteAccessor<Data<SetIndex> > tD2S(triangleD2S);
    helper::WriteAccessor<Data<SetIndex> > qS2D(quadS2D);
    helper::WriteAccessor<Data<SetIndex> > qD2S(quadD2S);
    helper::WriteAccessor<Data<SetIndex> > teS2D(tetrahedronS2D);
    helper::WriteAccessor<Data<SetIndex> > teD2S(tetrahedronD2S);
    helper::WriteAccessor<Data<SetIndex> > heS2D(hexahedronS2D);
    helper::WriteAccessor<Data<SetIndex> > heD2S(hexahedronD2S);
    int count = 0;
    while( itBegin != itEnd )
    {
        const TopologyChange* topoChange = *itBegin;
        TopologyChangeType changeType = topoChange->getChangeType();

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            sout << "[" << count << "]ENDING_EVENT" << sendl;
            toPointMod->propagateTopologicalChanges();
            toPointMod->notifyEndingEvent();
            toPointMod->propagateTopologicalChanges();
            if (!samePoints.getValue())
            {
                if (pS2D.size() != (unsigned)fromModel->getNbPoints()) serr << "Invalid pointS2D size : " << pS2D.size() << " != " << fromModel->getNbPoints() << sendl;
                if (pD2S.size() != (unsigned)  toModel->getNbPoints()) serr << "Invalid pointD2S size : " << pD2S.size() << " != " <<   toModel->getNbPoints() << sendl;
            }
            if (handleEdges.getValue())
            {
                if (eS2D.size() != (unsigned)fromModel->getNbEdges()) serr << "Invalid edgeS2D size : " << eS2D.size() << " != " << fromModel->getNbEdges() << sendl;
                if (eD2S.size() != (unsigned)  toModel->getNbEdges()) serr << "Invalid edgeD2S size : " << eD2S.size() << " != " <<   toModel->getNbEdges() << sendl;
            }
            if (handleTriangles.getValue())
            {
                if (tS2D.size() != (unsigned)fromModel->getNbTriangles()) serr << "Invalid triangleS2D size : " << tS2D.size() << " != " << fromModel->getNbTriangles() << sendl;
                if (tD2S.size() != (unsigned)  toModel->getNbTriangles()) serr << "Invalid triangleD2S size : " << tD2S.size() << " != " <<   toModel->getNbTriangles() << sendl;
            }
            if (handleQuads.getValue())
            {
                if (qS2D.size() != (unsigned)fromModel->getNbQuads()) serr << "Invalid quadS2D size : " << qS2D.size() << " != " << fromModel->getNbQuads() << sendl;
                if (qD2S.size() != (unsigned)  toModel->getNbQuads()) serr << "Invalid quadD2S size : " << qD2S.size() << " != " <<   toModel->getNbQuads() << sendl;
            }
            if (handleTetrahedra.getValue())
            {
                if (teS2D.size() != (unsigned)fromModel->getNbTetrahedra()) serr << "Invalid tetrahedronS2D size : " << teS2D.size() << " != " << fromModel->getNbTetrahedra() << sendl;
                if (teD2S.size() != (unsigned)  toModel->getNbTetrahedra()) serr << "Invalid tetrahedronD2S size : " << teD2S.size() << " != " <<   toModel->getNbTetrahedra() << sendl;
            }
            if (handleHexahedra.getValue())
            {
                if (heS2D.size() != (unsigned)fromModel->getNbHexahedra()) serr << "Invalid hexahedronS2D size : " << heS2D.size() << " != " << fromModel->getNbHexahedra() << sendl;
                if (heD2S.size() != (unsigned)  toModel->getNbHexahedra()) serr << "Invalid hexahedronD2S size : " << heD2S.size() << " != " <<   toModel->getNbHexahedra() << sendl;
            }
            if (!samePoints.getValue())
                sout << " P: "<<fromModel->getNbPoints() << "->" << toModel->getNbPoints() << "/" << (fromModel->getNbPoints() - toModel->getNbPoints());
            if (handleEdges.getValue())
                sout << " E: "<<fromModel->getNbEdges() << "->" << toModel->getNbEdges() << "/" << (fromModel->getNbEdges() - toModel->getNbEdges());
            if (handleTriangles.getValue())
                sout << " T: "<<fromModel->getNbTriangles() << "->" << toModel->getNbTriangles() << "/" << (fromModel->getNbTriangles() - toModel->getNbTriangles());
            if (handleQuads.getValue())
                sout << " Q: "<<fromModel->getNbQuads() << "->" << toModel->getNbQuads() << "/" << (fromModel->getNbQuads() - toModel->getNbQuads());
            if (handleTetrahedra.getValue())
                sout << " TE: "<<fromModel->getNbTetrahedra() << "->" << toModel->getNbTetrahedra() << "/" << (fromModel->getNbTetrahedra() - toModel->getNbTetrahedra());
            if (handleHexahedra.getValue())
                sout << " HE: "<<fromModel->getNbHexahedra() << "->" << toModel->getNbHexahedra() << "/" << (fromModel->getNbHexahedra() - toModel->getNbHexahedra());
            sout << sendl;
            break;
        }

        case core::topology::POINTSADDED:
        {
            const PointsAdded * pAdd = static_cast< const PointsAdded * >( topoChange );
            unsigned int pS0 = (samePoints.getValue() ? toModel->getNbPoints() : pS2D.size());
            unsigned int nSadd = pAdd->getNbAddedVertices();
            sout << "[" << count << "]POINTSADDED : " << nSadd << " : " << pS0 << " - " << (pS0 + nSadd-1) << sendl;
            if (samePoints.getValue())
            {
                toPointMod->addPointsProcess(pAdd->getNbAddedVertices());
                toPointMod->addPointsWarning(pAdd->getNbAddedVertices(), pAdd->ancestorsList, pAdd->coefs, true);
                toPointMod->propagateTopologicalChanges();
            }
            else
            {
                helper::vector< helper::vector<Index> > ancestors;
                helper::vector< helper::vector<double> > coefs;
                unsigned int nDadd = 0;
                unsigned int pD0 = pD2S.size();
                pS2D.resize(pS0+nSadd);
                ancestors.reserve(pAdd->ancestorsList.size());
                coefs.reserve(pAdd->coefs.size());
                for (Index pi = 0; pi < nSadd; ++pi)
                {
                    Index ps = pS0+pi;
                    pS2D[ps] = core::topology::Topology::InvalidID;
                    bool inDst = true;
                    if (!pAdd->ancestorsList.empty())
                        for (unsigned int i=0; i<pAdd->ancestorsList[pi].size() && inDst; ++i)
                        {
                            if (pAdd->coefs[pi][i] == 0.0) continue;
                            if (pS2D[pAdd->ancestorsList[pi][i]] == core::topology::Topology::InvalidID)
                                inDst = false;
                        }
                    if (!inDst) continue;
                    Index pd = pD0+nDadd;
                    pD2S.resize(pd+1);
                    pS2D[ps] = pd;
                    pD2S[pd] = ps;

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
                    sout << "    -> POINTSADDED : " << nDadd << " : " << pD0 << " - " << (pD0 + nDadd-1) << sendl;
                    toPointMod->addPointsProcess(nDadd);
                    toPointMod->addPointsWarning(nDadd, ancestors, coefs, true);
                    toPointMod->propagateTopologicalChanges();
                }
            }
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const PointsRemoved *pRem = static_cast< const PointsRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = pRem->getArray();
            if (samePoints.getValue())
            {
                sout << "[" << count << "]POINTSREMOVED : " << tab.size() << " : " << tab << sendl;
                toPointMod->removePointsWarning(tab, true);
                toPointMod->propagateTopologicalChanges();
                toPointMod->removePointsProcess(tab, true);
            }
            else
            {
                sofa::helper::vector<unsigned int> tab2;
                tab2.reserve(tab.size());
                for (unsigned int pi=0; pi<tab.size(); ++pi)
                {
                    Index ps = tab[pi];
                    Index pd = pS2D[ps];
                    if (pd == core::topology::Topology::InvalidID)
                        continue;
                    tab2.push_back(pd);
                }
                sout << "[" << count << "]POINTSREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2 << sendl;
                // apply removals in pS2D
                {
                    unsigned int last = pS2D.size() -1;
                    for (unsigned int i = 0; i < tab.size(); ++i)
                    {
                        Index ps = tab[i];
                        Index pd = pS2D[ps];
                        if (pd != core::topology::Topology::InvalidID)
                            pD2S[pd] = core::topology::Topology::InvalidID;
                        Index pd2 = pS2D[last];
                        pS2D[ps] = pd2;
                        if (pd2 != core::topology::Topology::InvalidID && pD2S[pd2] == last)
                            pD2S[pd2] = ps;
                        --last;
                    }
                    pS2D.resize( pS2D.size() - tab.size() );
                }
                if (!tab2.empty())
                {
                    toPointMod->removePointsWarning(tab2, true);
                    toPointMod->propagateTopologicalChanges();
                    toPointMod->removePointsProcess(tab2, true);
                    // apply removals in pD2S
                    {
                        unsigned int last = pD2S.size() -1;
                        for (unsigned int i = 0; i < tab2.size(); ++i)
                        {
                            Index pd = tab2[i];
                            Index ps = pD2S[pd];
                            if (ps != core::topology::Topology::InvalidID)
                                serr << "Invalid Point Remove" << sendl;
                            Index ps2 = pD2S[last];
                            pD2S[pd] = ps2;
                            if (ps2 != core::topology::Topology::InvalidID && pS2D[ps2] == last)
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
            const sofa::helper::vector<unsigned int> &tab = pRenumber->getIndexArray();
            const sofa::helper::vector<unsigned int> &inv_tab = pRenumber->getinv_IndexArray();
            if (samePoints.getValue())
            {
                sout << "[" << count << "]POINTSRENUMBERING : " << tab.size() << " : " << tab << sendl;
                toPointMod->renumberPointsWarning(tab, inv_tab, true);
                toPointMod->propagateTopologicalChanges();
                toPointMod->renumberPointsProcess(tab, inv_tab, true);
            }
            else
            {
                sofa::helper::vector<unsigned int> tab2;
                sofa::helper::vector<unsigned int> inv_tab2;
                tab2.resize(pD2S.size());
                inv_tab2.resize(pD2S.size());
                for (Index pd = 0; pd < pD2S.size(); ++pd)
                {
                    Index ps = pD2S[pd];
                    Index ps2 = (ps == core::topology::Topology::InvalidID) ? core::topology::Topology::InvalidID : tab[ps];
                    Index pd2 = (ps2 == core::topology::Topology::InvalidID) ? core::topology::Topology::InvalidID : pS2D[ps2];
                    if (pd2 == core::topology::Topology::InvalidID) pd2 = pd;
                    tab2[pd] = pd2;
                    inv_tab2[pd2] = pd;
                }
                sout << "[" << count << "]POINTSRENUMBERING : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2 << sendl;
                toPointMod->renumberPointsWarning(tab2, inv_tab2, true);
                toPointMod->propagateTopologicalChanges();
                toPointMod->renumberPointsProcess(tab2, inv_tab2, true);
                SetIndex pS2D0 = pS2D.ref();
                for (Index ps = 0; ps < pS2D.size(); ++ps)
                {
                    Index pd = pS2D0[tab2[ps]];
                    if (pd != core::topology::Topology::InvalidID)
                        pd = inv_tab2[pd];
                    pS2D[ps] = pd;
                }
                SetIndex pD2S0 = pD2S.ref();
                for (Index pd = 0; pd < pD2S.size(); ++pd)
                {
                    Index ps = pD2S0[tab[pd]];
                    if (ps != core::topology::Topology::InvalidID)
                        ps = inv_tab[ps];
                    pD2S[pd] = ps;
                }
            }
            break;
        }

        case core::topology::EDGESADDED:
        {
            if (!handleEdges.getValue()) break;
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesAdded *eAdd = static_cast< const EdgesAdded * >( topoChange );
            sout << "[" << count << "]EDGESADDED : " << eAdd->getNbAddedEdges() << " : " << eAdd->edgeIndexArray << " : " << eAdd->edgeArray << sendl;
            //toEdgeMod->addEdgesProcess(eAdd->edgeArray);
            //toEdgeMod->addEdgesWarning(eAdd->getNbAddedEdges(), eAdd->edgeArray, eAdd->edgeIndexArray, eAdd->ancestorsList, eAdd->coefs);
            //toEdgeMod->propagateTopologicalChanges();
            helper::vector< core::topology::BaseMeshTopology::Edge > edgeArray;
            helper::vector< Index > edgeIndexArray;
            helper::vector< helper::vector<Index> > ancestors;
            helper::vector< helper::vector<double> > coefs;
            unsigned int nSadd = eAdd->getNbAddedEdges();
            unsigned int nDadd = 0;
            unsigned int eS0 = eS2D.size();
            unsigned int eD0 = eD2S.size();
            eS2D.resize(eS0+nSadd);
            edgeArray.reserve(eAdd->edgeArray.size());
            edgeIndexArray.reserve(eAdd->edgeIndexArray.size());
            ancestors.reserve(eAdd->ancestorsList.size());
            coefs.reserve(eAdd->coefs.size());
            for (Index ei = 0; ei < nSadd; ++ei)
            {
                Index es = eAdd->edgeIndexArray[ei];
                eS2D[es] = core::topology::Topology::InvalidID;
                bool inDst = true;
                core::topology::BaseMeshTopology::Edge data = apply_map(eAdd->edgeArray[ei], pS2D);
                for (unsigned int i=0; i<data.size() && inDst; ++i)
                    if (data[i] == core::topology::Topology::InvalidID)
                        inDst = false;
                if (!eAdd->ancestorsList.empty())
                    for (unsigned int i=0; i<eAdd->ancestorsList[ei].size() && inDst; ++i)
                    {
                        if (eAdd->coefs[ei][i] == 0.0) continue;
                        if (eS2D[eAdd->ancestorsList[ei][i]] == core::topology::Topology::InvalidID)
                            inDst = false;
                    }
                if (!inDst) continue;
                Index ed = eD0+nDadd;
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
                sout << "    -> EDGESADDED : " << nDadd << " : " << edgeIndexArray << " : " << edgeArray << sendl;
                toEdgeMod->addEdgesProcess(edgeArray);
                toEdgeMod->addEdgesWarning(nDadd, edgeArray, edgeIndexArray, ancestors, coefs);
                toEdgeMod->propagateTopologicalChanges();
            }
            break;
        }

        case core::topology::EDGESREMOVED:
        {
            if (!handleEdges.getValue()) break;
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesRemoved *eRem = static_cast< const EdgesRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = eRem->getArray();
            //toEdgeMod->removeEdgesWarning(tab);
            //toEdgeMod->propagateTopologicalChanges();
            //toEdgeMod->removeEdgesProcess(tab, false);
            sofa::helper::vector<unsigned int> tab2;
            tab2.reserve(tab.size());
            for (unsigned int ei=0; ei<tab.size(); ++ei)
            {
                Index es = tab[ei];
                Index ed = eS2D[es];
                if (ed == core::topology::Topology::InvalidID)
                    continue;
                tab2.push_back(ed);
            }
            sout << "[" << count << "]EDGESREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2 << sendl;
            // apply removals in eS2D
            {
                unsigned int last = eS2D.size() -1;
                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    Index es = tab[i];
                    Index ed = eS2D[es];
                    if (ed != core::topology::Topology::InvalidID)
                        eD2S[ed] = core::topology::Topology::InvalidID;
                    Index ed2 = eS2D[last];
                    eS2D[es] = ed2;
                    if (ed2 != core::topology::Topology::InvalidID && eD2S[ed2] == last)
                        eD2S[ed2] = es;
                    --last;
                }
                eS2D.resize( eS2D.size() - tab.size() );
            }
            if (!tab2.empty())
            {
                toEdgeMod->removeEdgesWarning(tab2);
                toEdgeMod->propagateTopologicalChanges();
                toEdgeMod->removeEdgesProcess(tab2, false);
                // apply removals in eD2S
                {
                    unsigned int last = eD2S.size() -1;
                    for (unsigned int i = 0; i < tab2.size(); ++i)
                    {
                        Index ed = tab2[i];
                        Index es = eD2S[ed];
                        if (es != core::topology::Topology::InvalidID)
                            serr << "Invalid Edge Remove" << sendl;
                        Index es2 = eD2S[last];
                        eD2S[ed] = es2;
                        if (es2 != core::topology::Topology::InvalidID && eS2D[es2] == last)
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
            if (!handleTriangles.getValue()) break;
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesAdded *tAdd = static_cast< const TrianglesAdded * >( topoChange );
            sout << "[" << count << "]TRIANGLESADDED : " << tAdd->getNbAddedTriangles() << " : " << tAdd->triangleIndexArray << " : " << tAdd->triangleArray << sendl;
            if (!tAdd->ancestorsList.empty())
            {
                int count = 0;
                double sum = 0.0;
                sout << "   ";
                for (unsigned int i = 0; i < tAdd->ancestorsList.size(); ++i)
                {
                    sout << "    " << tAdd->ancestorsList[i];
                    count += tAdd->ancestorsList[i].size();
                    for (unsigned int j=0; j<tAdd->ancestorsList[i].size(); ++j)
                        sum += tAdd->coefs[i][j];
                }
                sout << sendl;
                sout << "     " << tAdd->ancestorsList.size() << " ancestor lists specified, " << count << " ancestors total, " << sum/tAdd->ancestorsList.size() << " avg coefs sum" << sendl;
            }
            //toTriangleMod->addTrianglesProcess(tAdd->triangleArray);
            //toTriangleMod->addTrianglesWarning(tAdd->getNbAddedTriangles(), tAdd->triangleArray, tAdd->triangleIndexArray, tAdd->ancestorsList, tAdd->coefs);
            //toTriangleMod->propagateTopologicalChanges();
            helper::vector< core::topology::BaseMeshTopology::Triangle > triangleArray;
            helper::vector< Index > triangleIndexArray;
            helper::vector< helper::vector<Index> > ancestors;
            helper::vector< helper::vector<double> > coefs;
            unsigned int nSadd = tAdd->getNbAddedTriangles();
            unsigned int nDadd = 0;
            unsigned int tS0 = tS2D.size();
            unsigned int tD0 = tD2S.size();
            tS2D.resize(tS0+nSadd);
            triangleArray.reserve(tAdd->triangleArray.size());
            triangleIndexArray.reserve(tAdd->triangleIndexArray.size());
            ancestors.reserve(tAdd->ancestorsList.size());
            coefs.reserve(tAdd->coefs.size());
            for (Index ti = 0; ti < nSadd; ++ti)
            {
                Index ts = tAdd->triangleIndexArray[ti];
                tS2D[ts] = core::topology::Topology::InvalidID;
                bool inDst = true;
                core::topology::BaseMeshTopology::Triangle data = apply_map(tAdd->triangleArray[ti], pS2D);
                for (unsigned int i=0; i<data.size() && inDst; ++i)
                    if (data[i] == core::topology::Topology::InvalidID)
                        inDst = false;
                if (!tAdd->ancestorsList.empty())
                    for (unsigned int i=0; i<tAdd->ancestorsList[ti].size() && inDst; ++i)
                    {
                        if (tAdd->coefs[ti][i] == 0.0) continue;
                        if (tS2D[tAdd->ancestorsList[ti][i]] == core::topology::Topology::InvalidID)
                            inDst = false;
                    }
                if (!inDst) continue;
                Index td = tD0+nDadd;
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
                sout << "    -> TRIANGLESADDED : " << nDadd  << " : " << triangleIndexArray << " : " << triangleArray << sendl;
                toTriangleMod->addTrianglesProcess(triangleArray);
                toTriangleMod->addTrianglesWarning(nDadd, triangleArray, triangleIndexArray, ancestors, coefs);
                toTriangleMod->propagateTopologicalChanges();
            }
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!handleTriangles.getValue()) break;
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesRemoved *tRem = static_cast< const TrianglesRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = tRem->getArray();
            //toTriangleMod->removeTrianglesWarning(tab);
            //toTriangleMod->propagateTopologicalChanges();
            //toTriangleMod->removeTrianglesProcess(tab, false);
            sofa::helper::vector<unsigned int> tab2;
            tab2.reserve(tab.size());
            for (unsigned int ti=0; ti<tab.size(); ++ti)
            {
                Index ts = tab[ti];
                Index td = tS2D[ts];
                if (td == core::topology::Topology::InvalidID)
                    continue;
                tab2.push_back(td);
            }
            sout << "[" << count << "]TRIANGLESREMOVED : " << tab.size() << " -> " << tab2.size() << " : " << tab << " -> " << tab2 << sendl;
            // apply removals in tS2D
            {
                unsigned int last = tS2D.size() -1;
                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    Index ts = tab[i];
                    Index td = tS2D[ts];
                    if (td != core::topology::Topology::InvalidID)
                        tD2S[td] = core::topology::Topology::InvalidID;
                    Index td2 = tS2D[last];
                    tS2D[ts] = td2;
                    if (td2 != core::topology::Topology::InvalidID && tD2S[td2] == last)
                        tD2S[td2] = ts;
                    --last;
                }
                tS2D.resize( tS2D.size() - tab.size() );
            }
            if (!tab2.empty())
            {
                toTriangleMod->removeTrianglesWarning(tab2);
                toTriangleMod->propagateTopologicalChanges();
                toTriangleMod->removeTrianglesProcess(tab2, !handleEdges.getValue());
                // apply removals in tD2S
                {
                    unsigned int last = tD2S.size() -1;
                    for (unsigned int i = 0; i < tab2.size(); ++i)
                    {
                        Index td = tab2[i];
                        Index ts = tD2S[td];
                        if (ts != core::topology::Topology::InvalidID)
                            serr << "Invalid Triangle Remove" << sendl;
                        Index ts2 = tD2S[last];
                        tD2S[td] = ts2;
                        if (ts2 != core::topology::Topology::InvalidID && tS2D[ts2] == last)
                            tS2D[ts2] = td;
                        --last;
                    }
                    tD2S.resize( tD2S.size() - tab2.size() );
                }
            }
            break;
        }

        default:
            serr << "Unknown topological change " << changeType << sendl;
            break;
        };
        ++count;
        ++itBegin;
    }
    toPointMod->propagateTopologicalChanges();
}

} // namespace topology

} // namespace component

} // namespace sofa

