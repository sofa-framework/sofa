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
#define SOFA_COMPONENT_COLLISION_TETRAHEDRONCOLLISIONMODEL_CPP
#include <SofaMiscCollision/TetrahedronModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>
#include <iostream>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

int TetrahedronCollisionModelClass = core::RegisterObject("collision model using a tetrahedral mesh, as described in BaseMeshTopology")
        .add< TetrahedronCollisionModel >()
        .addAlias("Tetrahedron")
        .addAlias("TetrahedronModel")
        ;

TetrahedronCollisionModel::TetrahedronCollisionModel()
    : tetra(nullptr)
    , mstate(nullptr)
    , m_topology(nullptr)
    , m_topologyRevision(-1)
    , l_topology(initLink("topology", "link to the topology container"))
{
    enum_type = TETRAHEDRON_TYPE;
}

void TetrahedronCollisionModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
    if (getPrevious() != nullptr) getPrevious()->resize(0); // force recomputation of bounding tree
}

void TetrahedronCollisionModel::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (!m_topology)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << ". TetrahedronCollisionModel requires a BaseMeshTopology";
        sofa::core::objectmodel::BaseObject::d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==nullptr)
    {
        msg_error() << "TetrahedronCollisionModel requires a Vec3 Mechanical Model";
        return;
    }

    updateFromTopology();
}


void TetrahedronCollisionModel::updateFromTopology()
{
    int revision = m_topology->getRevision();
    if (revision == m_topologyRevision)
        return;
    
    if (m_topology->getNbTetrahedra() != elems.size())
    {
        tetra = &m_topology->getTetrahedra();
        resize(tetra->size());
    }

    m_topologyRevision = revision;
}


void TetrahedronCollisionModel::addTetraToDraw(const Tetrahedron& t, std::vector<sofa::defaulttype::Vector3>& tetraVertices, std::vector<sofa::defaulttype::Vector3>& normalVertices)
{
    Coord p1 = t.p1();
    Coord p2 = t.p2();
    Coord p3 = t.p3();
    Coord p4 = t.p4();
    Coord c = (p1 + p2 + p3 + p4)*0.25f;
    auto computeP = [](Coord center, Coord pos) {pos += (center - pos)*0.1f; };
    computeP(c, p1);
    computeP(c, p2);
    computeP(c, p3);
    computeP(c, p4);

    Coord n1, n2, n3, n4;
    auto computeN = [](Coord normal, Coord pos1, Coord pos2, Coord pos3) {normal = cross(pos3 - pos1, pos2 - pos1); normal.normalize(); };
    computeN(n1,p1,p2,p3);
    computeN(n1,p1,p3,p4);
    computeN(n1,p1,p4,p2);
    computeN(n1,p2,p4,p3);

    tetraVertices.push_back(p1);
    tetraVertices.push_back(p2);
    tetraVertices.push_back(p3);
    tetraVertices.push_back(p4);

    Coord p;
    auto updateP = [](Coord pPos, Coord pos1, Coord pos2, Coord pos3) {pPos = (pos1 + pos2 + pos3)*(1.0 / 3.0); };
    updateP(p,p1,p2,p3);
    normalVertices.push_back(p);
    normalVertices.push_back(p + n1*0.1);
    updateP(p,p1,p3,p4);
    normalVertices.push_back(p);
    normalVertices.push_back(p + n2*0.1);
    updateP(p,p1,p4,p2);
    normalVertices.push_back(p);
    normalVertices.push_back(p + n3*0.1);
    updateP(p,p2,p3,p4);
    normalVertices.push_back(p);
    normalVertices.push_back(p + n4*0.1);
}

void TetrahedronCollisionModel::draw(const core::visual::VisualParams* vparams,int index)
{
    vparams->drawTool()->saveLastState();

    std::vector<sofa::defaulttype::Vector3> tetraVertices;
    std::vector<sofa::defaulttype::Vector3> normalVertices;

    Tetrahedron t(this, index);
    this->addTetraToDraw(t, tetraVertices, normalVertices);

    vparams->drawTool()->drawTetrahedra(tetraVertices, defaulttype::Vec<4, float>(getColor4f()));

    vparams->drawTool()->restoreLastState();
}

void TetrahedronCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->saveLastState();
    if (mstate && m_topology && vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);

        vparams->drawTool()->enableLighting();
        vparams->drawTool()->setMaterial(defaulttype::Vec<4, float>(getColor4f()));

        std::vector<sofa::defaulttype::Vector3> tetraVertices;
        std::vector<sofa::defaulttype::Vector3> normalVertices;
        for (int i = 0; i<size; i++)
        {
            Tetrahedron t(this, i);
            this->addTetraToDraw(t, tetraVertices, normalVertices);
        }

        vparams->drawTool()->drawTetrahedra(tetraVertices, defaulttype::Vec<4, float>(getColor4f()));
        if (vparams->displayFlags().getShowNormals())
        {
            vparams->drawTool()->drawLines(normalVertices, 1.0, defaulttype::Vec<4, float>(getColor4f()));
        }

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
    }
    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->restoreLastState();
}

void TetrahedronCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    if (!mstate || !m_topology) return;
    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateFromTopology();

    Vector3 minElem, maxElem;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    for (int i=0; i<size; i++)
    {
        Tetrahedron t(this,i);
        const Vector3& pt1 = x[t.p1Index()];
        const Vector3& pt2 = x[t.p2Index()];
        const Vector3& pt3 = x[t.p3Index()];
        const Vector3& pt4 = x[t.p4Index()];
        Matrix3 m, minv;
        m[0] = pt2-pt1;
        m[1] = pt3-pt1;
        m[2] = pt4-pt1;
        m.transpose();
        minv.invert(m);
        elems[i].coord0 = pt1;
        elems[i].bary2coord = m;
        elems[i].coord2bary = minv;
    }

    if (maxDepth == 0)
    {
        // no hierarchy
        if (empty())
            cubeModel->resize(0);
        else
        {
            cubeModel->resize(1);
            minElem = x[0];
            maxElem = x[0];
            for (unsigned i=1; i<x.size(); i++)
            {
                const Vector3& pt1 = x[i];
                if (pt1[0] > maxElem[0]) maxElem[0] = pt1[0];
                else if (pt1[0] < minElem[0]) minElem[0] = pt1[0];
                if (pt1[1] > maxElem[1]) maxElem[1] = pt1[1];
                else if (pt1[1] < minElem[1]) minElem[1] = pt1[1];
                if (pt1[2] > maxElem[2]) maxElem[2] = pt1[2];
                else if (pt1[2] < minElem[2]) minElem[2] = pt1[2];
            }
            cubeModel->setLeafCube(0, std::make_pair(this->begin(),this->end()), minElem, maxElem); // define the bounding box of the current Tetrahedron
        }
    }
    else
    {
        cubeModel->resize(size);  // size = number of Tetrahedrons
        if (!empty())
        {
            for (int i=0; i<size; i++)
            {
                Tetrahedron t(this,i);
                const Vector3& pt1 = x[t.p1Index()];
                const Vector3& pt2 = x[t.p2Index()];
                const Vector3& pt3 = x[t.p3Index()];
                const Vector3& pt4 = x[t.p4Index()];
                for (int c = 0; c < 3; c++)
                {
                    minElem[c] = pt1[c];
                    maxElem[c] = pt1[c];
                    if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                    else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                    if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                    else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
                    if (pt4[c] > maxElem[c]) maxElem[c] = pt4[c];
                    else if (pt4[c] < minElem[c]) minElem[c] = pt4[c];
                }
                cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current Tetrahedron
            }
            cubeModel->computeBoundingTree(maxDepth);
        }
    }
}

ContactMapperCreator< ContactMapper<TetrahedronCollisionModel> > TetrahedronContactMapperClass("default",true);

template class SOFA_MISC_COLLISION_API ContactMapper<TetrahedronCollisionModel, sofa::defaulttype::Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa
