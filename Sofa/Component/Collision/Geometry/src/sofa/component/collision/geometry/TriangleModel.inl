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
#pragma once
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/TopologyChange.h>
#include <vector>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
TriangleCollisionModel<DataTypes>::TriangleCollisionModel()
    : d_bothSide(initData(&d_bothSide, false, "bothSide", "activate collision on both side of the triangle model") )
    , d_computeNormals(initData(&d_computeNormals, true, "computeNormals", "set to false to disable computation of triangles normal"))
    , d_useCurvature(initData(&d_useCurvature, false, "useCurvature", "use the curvature of the mesh to avoid some self-intersection test"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_mstate(nullptr)
    , m_topology(nullptr)
    , m_needsUpdate(true)
    , m_topologyRevision(-1)
    , m_pointModels(nullptr)
{
    m_triangles = &m_internalTriangles;
    enum_type = TRIANGLE_TYPE;
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::resize(sofa::Size size)
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
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
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
        msg_error() << "No MechanicalState found. TriangleCollisionModel<sofa::defaulttype::Vec3Types> requires a Vec3 MechanicalState in the same Node.";
        modelsOk = false;
    }

    if (!modelsOk)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
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
    for (sofa::Size i=0; i<size; i++)
    {
        Element t(this,i);
        const type::Vec3& pt1 = t.p1();
        const type::Vec3& pt2 = t.p2();
        const type::Vec3& pt3 = t.p3();

        t.n() = cross(pt2-pt1,pt3-pt1);
        t.n().normalize();
    }
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::updateFromTopology()
{
    const int revision = m_topology->getRevision();
    if (revision == m_topologyRevision)
        return;

    m_topologyRevision = revision;

    const sofa::Size nquads = m_topology->getNbQuads();
    const sofa::Size ntris = m_topology->getNbTriangles();

    if (nquads == 0) // only triangles
    {
        resize(ntris);
        m_triangles = &m_topology->getTriangles();
    }
    else
    {
        const sofa::Size newsize = ntris+2*nquads;
        const sofa::Size npoints = m_mstate->getSize();

        m_triangles = &m_internalTriangles;
        m_internalTriangles.resize(newsize);
        resize(newsize);

        sofa::Index index = 0;
        for (sofa::Index i=0; i<ntris; i++)
        {
            core::topology::BaseMeshTopology::Triangle idx = m_topology->getTriangle(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
            {
                msg_error() << "Vertex index out of range in triangle " << i << ": " << idx[0] << " " << idx[1] << " " << idx[2] << " ( total points=" << npoints << ")";
                if (idx[0] >= npoints) idx[0] = npoints - 1;
                if (idx[1] >= npoints) idx[1] = npoints - 1;
                if (idx[2] >= npoints) idx[2] = npoints - 1;
            }
            m_internalTriangles[index] = idx;
            ++index;
        }
        for (sofa::Index i=0; i<nquads; i++)
        {
            core::topology::BaseMeshTopology::Quad idx = m_topology->getQuad(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
            {
                msg_error() << "Vertex index out of range in quad " << i << ": " << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << " ( total points=" << npoints << ")";
                if (idx[0] >= npoints) idx[0] = npoints - 1;
                if (idx[1] >= npoints) idx[1] = npoints - 1;
                if (idx[2] >= npoints) idx[2] = npoints - 1;
                if (idx[3] >= npoints) idx[3] = npoints - 1;
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
bool TriangleCollisionModel<DataTypes>::canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2)
{
    if (!this->bSelfCollision.getValue()) return true; // we need to perform this verification process only for the selfcollision case.
    if (this->getContext() != model2->getContext()) return true;

    Element t(this,index);
    if (model2 == m_pointModels)
    {
        // if point belong to the triangle, return false
        if (index2 == t.p1Index() || index2 == t.p2Index() || index2 == t.p3Index())
            return false;

        //// TODO : case with auto-collis with segment and auto-collis with itself
    }

    return true;

}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();

    // check first that topology didn't changed
    if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();

    if (m_needsUpdate && !cubeModel->empty())
        cubeModel->resize(0);

    if (!isMoving() && !cubeModel->empty() && !m_needsUpdate)
        return; // No need to recompute BBox if immobile nor if mesh didn't change.

    // set to false to avoid excesive loop
    m_needsUpdate=false;

    type::Vec3 minElem, maxElem;
    const VecCoord& x = this->m_mstate->read(core::ConstVecCoordId::position())->getValue();

    const bool calcNormals = d_computeNormals.getValue();

    cubeModel->resize(size);  // size = number of triangles
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            Element t(this,i);

            const type::Vec3& pt1 = x[t.p1Index()];
            const type::Vec3& pt2 = x[t.p2Index()];
            const type::Vec3& pt3 = x[t.p3Index()];

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

            if(d_useCurvature.getValue())
                cubeModel->setParentOf(i, minElem, maxElem, t.n()); // define the bounding box of the current triangle
            else
                cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
void TriangleCollisionModel<DataTypes>::computeContinuousBoundingTree(SReal dt, int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();

    // check first that topology didn't changed
    if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();

    if (m_needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !m_needsUpdate) return; // No need to recompute BBox if immobile nor if mesh didn't change.

    m_needsUpdate=false;
    type::Vec3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            Element t(this,i);
            const type::Vec3& pt1 = t.p1();
            const type::Vec3& pt2 = t.p2();
            const type::Vec3& pt3 = t.p3();
            const type::Vec3 pt1v = pt1 + t.v1()*dt;
            const type::Vec3 pt2v = pt2 + t.v2()*dt;
            const type::Vec3 pt3v = pt3 + t.v3()*dt;

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

            if(d_useCurvature.getValue())
                cubeModel->setParentOf(i, minElem, maxElem, t.n(), acos(cross(pt2v-pt1v,pt3v-pt1v).normalized() * t.n()));
            else
                cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
int TriangleCollisionModel<DataTypes>::getTriangleFlags(Topology::TriangleID i)
{
    int f = 0;
    sofa::core::topology::BaseMeshTopology::Triangle t = (*m_triangles)[i];

    if (i < m_topology->getNbTriangles())
    {
        for (sofa::Index j=0; j<3; ++j)
        {
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex& tav = m_topology->getTrianglesAroundVertex(t[j]);
            if (tav[0] == (sofa::core::topology::BaseMeshTopology::TriangleID)i)
            {
                f |= (FLAG_P1 << j);
            }
        }

        const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e = m_topology->getEdgesInTriangle(i);

        for (sofa::Index j=0; j<3; ++j)
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

    static constexpr Real max_real = std::numeric_limits<Real>::max();
    static constexpr Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    const auto& positions = this->m_mstate->read(core::ConstVecCoordId::position())->getValue();

    for (sofa::Size i=0; i<size; i++)
    {
        const type::Vec3& pt1 = positions[(*this->m_triangles)[i][0]];
        const type::Vec3& pt2 = positions[(*this->m_triangles)[i][1]];
        const type::Vec3& pt3 = positions[(*this->m_triangles)[i][2]];

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

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox,maxBBox));
}


template<class DataTypes>
void TriangleCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams , sofa::Index index)
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

        std::vector< type::Vec3 > points;
        std::vector< type::Vec<3,int> > indices;
        std::vector< type::Vec3 > normals;
        int index=0;
        for (sofa::Size i=0; i<size; i++)
        {
            Element t(this,i);
            normals.push_back(t.n());
            points.push_back(t.p1());
            points.push_back(t.p2());
            points.push_back(t.p3());
            indices.push_back(type::Vec<3,int>(index,index+1,index+2));
            index+=3;
        }

        vparams->drawTool()->setLightingEnabled(true);
        const auto c = getColor4f();
        vparams->drawTool()->drawTriangles(points, indices, normals, sofa::type::RGBAColor(c[0], c[1], c[2], c[3]));
        vparams->drawTool()->setLightingEnabled(false);
        vparams->drawTool()->setPolygonMode(0,false);


        if (vparams->displayFlags().getShowNormals())
        {
            std::vector< type::Vec3 > points;
            for (sofa::Size i=0; i<size; i++)
            {
                Element t(this,i);
                points.push_back((t.p1()+t.p2()+t.p3())/3.0);
                points.push_back(points.back()+t.n());
            }

            vparams->drawTool()->drawLines(points, 1, sofa::type::RGBAColor::white());

        }
    }
    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}

template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p1() const { return this->model->m_mstate->read(core::ConstVecCoordId::position())->getValue()[(*(this->model->m_triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p2() const { return this->model->m_mstate->read(core::ConstVecCoordId::position())->getValue()[(*(this->model->m_triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p3() const { return this->model->m_mstate->read(core::ConstVecCoordId::position())->getValue()[(*(this->model->m_triangles))[this->index][2]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p(Index i) const {
    return this->model->m_mstate->read(core::ConstVecCoordId::position())->getValue()[(*(this->model->m_triangles))[this->index][i]];
}
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::operator[](Index i) const {
    return this->model->m_mstate->read(core::ConstVecCoordId::position())->getValue()[(*(this->model->m_triangles))[this->index][i]];
}

template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p1Free() const { return (this->model->m_mstate->read(sofa::core::ConstVecCoordId::freePosition())->getValue())[(*(this->model->m_triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p2Free() const { return (this->model->m_mstate->read(sofa::core::ConstVecCoordId::freePosition())->getValue())[((*this->model->m_triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p3Free() const { return (this->model->m_mstate->read(sofa::core::ConstVecCoordId::freePosition())->getValue())[(*(this->model->m_triangles))[this->index][2]]; }

template<class DataTypes>
inline typename TTriangle<DataTypes>::Index TTriangle<DataTypes>::p1Index() const { return (*(this->model->m_triangles))[this->index][0]; }
template<class DataTypes>
inline typename TTriangle<DataTypes>::Index TTriangle<DataTypes>::p2Index() const { return (*(this->model->m_triangles))[this->index][1]; }
template<class DataTypes>
inline typename TTriangle<DataTypes>::Index TTriangle<DataTypes>::p3Index() const { return (*(this->model->m_triangles))[this->index][2]; }

template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v1() const { return (this->model->m_mstate->read(core::ConstVecDerivId::velocity())->getValue())[(*(this->model->m_triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v2() const { return this->model->m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(this->model->m_triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v3() const { return this->model->m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(this->model->m_triangles))[this->index][2]]; }
template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v(Index i) const { return this->model->m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(this->model->m_triangles))[this->index][i]]; }

template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::n() const { return this->model->m_normals[this->index]; }
template<class DataTypes>
inline       typename DataTypes::Deriv& TTriangle<DataTypes>::n()       { return this->model->m_normals[this->index]; }

template<class DataTypes>
inline int TTriangle<DataTypes>::flags() const { return this->model->getTriangleFlags(this->index); }

template<class DataTypes>
inline bool TTriangle<DataTypes>::hasFreePosition() const { return this->model->m_mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }

template<class DataTypes>
inline typename DataTypes::Deriv TriangleCollisionModel<DataTypes>::velocity(sofa::Index index) const { return (m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(m_triangles))[index][0]] + m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(m_triangles))[index][1]] +
                                                                                                m_mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(m_triangles))[index][2]])/((Real)(3.0)); }


} //namespace sofa::component::collision::geometry
