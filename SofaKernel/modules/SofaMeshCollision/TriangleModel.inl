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
//#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_INL
//#define SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_INL

#include <SofaMeshCollision/TriangleModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/Triangle.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <sofa/core/CollisionElement.h>
#include <vector>
#include <iostream>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

template<class DataTypes>
TriangleCollisionModel<DataTypes>::TriangleCollisionModel()
    : d_bothSide(initData(&d_bothSide, false, "bothSide", "activate collision on both side of the triangle model") )
    , d_computeNormals(initData(&d_computeNormals, true, "computeNormals", "set to false to disable computation of triangles normal"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_mstate(nullptr)
    , m_topology(nullptr)
    , m_needsUpdate(true)
    , m_topologyRevision(-1)
    , m_pointModels(nullptr)
    , m_lmdFilter(nullptr)
{
    m_triangles = &m_internalTriangles;
    enum_type = TRIANGLE_TYPE;
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);
    m_normals.resize(size);
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::init()
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
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << ". TriangleCollisionModel<sofa::defaulttype::Vec3Types> requires a Triangular Topology";
        sofa::core::objectmodel::BaseObject::d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // TODO epernod 2019-01-21: Check if this call super is needed.
    this->CollisionModel::init();
    m_mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (this->getContext()->getMechanicalState());

    this->getContext()->get(m_pointModels);

    // Check object pointer access
    bool modelsOk = true;
    if (m_mstate == nullptr)
    {
        msg_error() << "No MechanicalObject found. TriangleCollisionModel<sofa::defaulttype::Vec3Types> requires a Vec3 Mechanical Model in the same Node.";
        modelsOk = false;
    }

    if (!modelsOk)
    {
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }

    simulation::Node* node = dynamic_cast< simulation::Node* >(this->getContext());
    if (node != 0)
    {
        m_lmdFilter = node->getNodeObject< TriangleLocalMinDistanceFilter >();
    }

    // check if topology is using triangles and quads at the same time.
    if (m_topology->getNbQuads() != 0)
    {
        updateFromTopology(); // in this case, need to create a single buffer with both topology
        // updateNormals will be call in updateFromTopology
    }
    else
    {
        // just redirect to the topology buffer.
        m_triangles = &m_topology->getTriangles();
        resize(m_topology->getNbTriangles());
        updateNormals();
    }
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::updateNormals()
{
    for (int i=0; i<size; i++)
    {
        Element t(this,i);
        const defaulttype::Vector3& pt1 = t.p1();
        const defaulttype::Vector3& pt2 = t.p2();
        const defaulttype::Vector3& pt3 = t.p3();

        t.n() = cross(pt2-pt1,pt3-pt1);
        t.n().normalize();
    }
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::updateFromTopology()
{
    int revision = m_topology->getRevision();
    if (revision == m_topologyRevision)
        return;

    m_topologyRevision = revision;

    const unsigned nquads = m_topology->getNbQuads();
    const unsigned ntris = m_topology->getNbTriangles();

    if (nquads == 0) // only triangles
    {
        resize(ntris);
        m_triangles = & m_topology->getTriangles();
    }
    else
    {
        const unsigned newsize = ntris+2*nquads;
        const unsigned npoints = m_mstate->getSize();

        m_triangles = &m_internalTriangles;
        m_internalTriangles.resize(newsize);
        resize(newsize);

        int index = 0;
        for (unsigned i=0; i<ntris; i++)
        {
            core::topology::BaseMeshTopology::Triangle idx = m_topology->getTriangle(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
            {
                msg_error() << "Vertex index out of range in triangle " << i << ": " << idx[0] << " " << idx[1] << " " << idx[2] <<" ( total points=" << npoints << ")";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
            }
            m_internalTriangles[index] = idx;
            ++index;
        }
        for (unsigned i=0; i<nquads; i++)
        {
            core::topology::BaseMeshTopology::Quad idx = m_topology->getQuad(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
            {
                msg_error() << "Vertex index out of range in quad " << i << ": " << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << " ( total points=" << npoints << ")";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
                if (idx[3] >= npoints) idx[3] = npoints-1;
            }
            m_internalTriangles[index][0] = idx[1];
            m_internalTriangles[index][1] = idx[2];
            m_internalTriangles[index][2] = idx[0];
            ++index;
            m_internalTriangles[index][0] = idx[3];
            m_internalTriangles[index][1] = idx[0];
            m_internalTriangles[index][2] = idx[2];
            ++index;
        }
    }
    updateNormals();

    // topology has changed, force boudingTree recomputation
    m_needsUpdate = true;
}


template<class DataTypes>
bool TriangleCollisionModel<DataTypes>::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    if (!this->bSelfCollision.getValue()) return true; // we need to perform this verification process only for the selfcollision case.
    if (this->getContext() != model2->getContext()) return true;

    Element t(this,index);
    if (model2 == m_pointModels)
    {
        // if point belong to the triangle, return false
        if ( index2==t.p1Index() || index2==t.p2Index() || index2==t.p3Index())
            return false;

        //// if the point belong to the the neighborhood of the triangle, return false
        //for (unsigned int i1=0; i1<EdgesAroundVertex11.size(); i1++)
        //{
        //	unsigned int e11 = EdgesAroundVertex11[i1];
        //	p11 = elems[e11].i1;
        //	p12 = elems[e11].i2;
        //	if (index2==p11 || index2==p12)
        //		return false;
        //}
        //for (unsigned int i1=0; i1<EdgesAroundVertex11.size(); i1++)
        //{
        //	unsigned int e12 = EdgesAroundVertex12[i1];
        //	p11 = elems[e12].i1;
        //	p12 = elems[e12].i2;
        //	if (index2==p11 || index2==p12)
        //		return false;
    }

    //// TODO : case with auto-collis with segment and auto-collis with itself

    return true;

}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();

    // check first that topology didn't changed
    if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();

    if (m_needsUpdate && !cubeModel->empty()) cubeModel->resize(0);

    if (!isMoving() && !cubeModel->empty() && !m_needsUpdate) return; // No need to recompute BBox if immobile nor if mesh didn't change.

    // set to false to avoid excesive loop
    m_needsUpdate=false;

    defaulttype::Vector3 minElem, maxElem;
    const VecCoord& x = this->m_mstate->read(core::ConstVecCoordId::position())->getValue();

    const bool calcNormals = d_computeNormals.getValue();

    cubeModel->resize(size);  // size = number of triangles
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            Element t(this,i);

            const defaulttype::Vector3& pt1 = x[t.p1Index()];
            const defaulttype::Vector3& pt2 = x[t.p2Index()];
            const defaulttype::Vector3& pt3 = x[t.p3Index()];

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }
            if (calcNormals)
            {
                // Also recompute normal vector
                t.n() = cross(pt2-pt1,pt3-pt1);
                t.n().normalize();
            }
            cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current triangle
        }
        cubeModel->computeBoundingTree(maxDepth);
    }


    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();

    // check first that topology didn't changed
    if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();

    if (m_needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !m_needsUpdate) return; // No need to recompute BBox if immobile nor if mesh didn't change.

    m_needsUpdate=false;
    defaulttype::Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            Element t(this,i);
            const defaulttype::Vector3& pt1 = t.p1();
            const defaulttype::Vector3& pt2 = t.p2();
            const defaulttype::Vector3& pt3 = t.p3();
            const defaulttype::Vector3 pt1v = pt1 + t.v1()*dt;
            const defaulttype::Vector3 pt2v = pt2 + t.v2()*dt;
            const defaulttype::Vector3 pt3v = pt3 + t.v3()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
                if (pt3v[c] > maxElem[c]) maxElem[c] = pt3v[c];
                else if (pt3v[c] < minElem[c]) minElem[c] = pt3v[c];

                minElem[c] -= distance;
                maxElem[c] += distance;
            }

            // Also recompute normal vector
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
TriangleLocalMinDistanceFilter *TriangleCollisionModel<DataTypes>::getFilter() const
{
    return m_lmdFilter;
}


template<class DataTypes>
void TriangleCollisionModel<DataTypes>::setFilter(TriangleLocalMinDistanceFilter *lmdFilter)
{
    m_lmdFilter = lmdFilter;
}

template<class DataTypes>
int TriangleCollisionModel<DataTypes>::getTriangleFlags(Topology::TriangleID i)
{
    int f = 0;
    sofa::core::topology::BaseMeshTopology::Triangle t = (*m_triangles)[i];

    if (i < m_topology->getNbTriangles())
    {
        for (unsigned int j=0; j<3; ++j)
        {
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex& tav = m_topology->getTrianglesAroundVertex(t[j]);
            if (tav[0] == (sofa::core::topology::BaseMeshTopology::TriangleID)i)
                f |= (FLAG_P1 << j);
        }

        const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e = m_topology->getEdgesInTriangle(i);

        for (unsigned int j=0; j<3; ++j)
        {
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& tae = m_topology->getTrianglesAroundEdge(e[j]);
            if (tae[0] == (sofa::core::topology::BaseMeshTopology::TriangleID)i)
                f |= (FLAG_E23 << j);
            if (tae.size() == 1)
                f |= (FLAG_BE23 << j);
        }
    }
    else
    {
        /// \todo flags for quads
    }
    return f;
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    // check first that topology didn't changed
    if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (int i=0; i<size; i++)
    {
        Element t(this,i);
        const defaulttype::Vector3& pt1 = t.p1();
        const defaulttype::Vector3& pt2 = t.p2();
        const defaulttype::Vector3& pt3 = t.p3();

        for (int c=0; c<3; c++)
        {
            if (pt1[c] > maxBBox[c]) maxBBox[c] = (Real)pt1[c];
            else if (pt1[c] < minBBox[c]) minBBox[c] = (Real)pt1[c];

            if (pt2[c] > maxBBox[c]) maxBBox[c] = (Real)pt2[c];
            else if (pt2[c] < minBBox[c]) minBBox[c] = (Real)pt2[c];

            if (pt3[c] > maxBBox[c]) maxBBox[c] = (Real)pt3[c];
            else if (pt3[c] < minBBox[c]) minBBox[c] = (Real)pt3[c];
        }
    }

    this->f_bbox.setValue(sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}


template<class DataTypes>
void TriangleCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams ,int index)
{
    Element t(this,index);

    vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawTriangle( t.p1(), t.p2(), t.p3(), t.n() );
    vparams->drawTool()->setLightingEnabled(false);
}


template<class DataTypes>
void TriangleCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        // In case topology has changed but drawing is called before the updateFromTopology has been computed, just exit to avoid computation in drawing thread.
        if (m_topology->getRevision() != m_topologyRevision)
            return;

        if (d_bothSide.getValue() || vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());
        else
        {
            vparams->drawTool()->setPolygonMode(2,true);
            vparams->drawTool()->setPolygonMode(1,false);
        }

        std::vector< defaulttype::Vector3 > points;
        std::vector< defaulttype::Vec<3,int> > indices;
        std::vector< defaulttype::Vector3 > normals;
        int index=0;
        for (int i=0; i<size; i++)
        {
            Element t(this,i);
            normals.push_back(t.n());
            points.push_back(t.p1());
            points.push_back(t.p2());
            points.push_back(t.p3());
            indices.push_back(defaulttype::Vec<3,int>(index,index+1,index+2));
            index+=3;
        }

        vparams->drawTool()->setLightingEnabled(true);
        vparams->drawTool()->drawTriangles(points, indices, normals, defaulttype::Vec<4,float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false);
        vparams->drawTool()->setPolygonMode(0,false);


        if (vparams->displayFlags().getShowNormals())
        {
            std::vector< defaulttype::Vector3 > points;
            for (int i=0; i<size; i++)
            {
                Element t(this,i);
                points.push_back((t.p1()+t.p2()+t.p3())/3.0);
                points.push_back(points.back()+t.n());
            }

            vparams->drawTool()->drawLines(points, 1, defaulttype::Vec<4,float>(1,1,1,1));

        }
    }
    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}



} // namespace collision

} // namespace component

} // namespace sofa

//#endif
