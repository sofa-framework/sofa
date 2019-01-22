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
TTriangleModel<DataTypes>::TTriangleModel()
    : bothSide(initData(&bothSide, false, "bothSide", "activate collision on both side of the triangle model") )
    , computeNormals(initData(&computeNormals, true, "computeNormals", "set to false to disable computation of triangles normal"))
    , m_mstate(NULL)
    , m_topology(NULL)
    , needsUpdate(true)
    , m_topologyRevision(-1)
    , m_pointModels(NULL)
    , m_lmdFilter(NULL)
{
    p_triangles = &my_triangles;
    enum_type = TRIANGLE_TYPE;
}

template<class DataTypes>
void TTriangleModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);
    normals.resize(size);
}

template<class DataTypes>
void TTriangleModel<DataTypes>::init()
{
    m_topology = this->getContext()->getMeshTopology();

    // TODO epernod 2019-01-21: Check if this call super is needed.
    this->CollisionModel::init();
    m_mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (this->getContext()->getMechanicalState());

    this->getContext()->get(m_pointModels);

    if (m_mstate == NULL)
    {
        msg_error() << "No MechanicalObject found. TriangleModel requires a Vec3 Mechanical Model in the same Node.";
        return;
    }

    if (m_topology == NULL)
    {
        msg_error() << "No Topology found. TriangleModel requires a Triangular Topology in the same Node.";
        return;
    }

    simulation::Node* node = dynamic_cast< simulation::Node* >(this->getContext());
    if (node != 0)
    {
        m_lmdFilter = node->getNodeObject< TriangleLocalMinDistanceFilter >();
    }


    p_triangles = &m_topology->getTriangles();
    resize(m_topology->getNbTriangles());

    updateFromTopology();
    updateNormals();
}

template<class DataTypes>
void TTriangleModel<DataTypes>::updateNormals()
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
void TTriangleModel<DataTypes>::updateFromTopology()
{
    //    needsUpdate = false;
    const unsigned npoints = m_mstate->getSize();
    const unsigned ntris = m_topology->getNbTriangles();
    const unsigned nquads = m_topology->getNbQuads();
    const unsigned newsize = ntris+2*nquads;

    int revision = m_topology->getRevision();
    if (newsize==(unsigned)size && revision == meshRevision)
        return;
    meshRevision = revision;

    needsUpdate=true;

    resize(newsize);

    if (newsize == ntris)
    {
        // no need to copy the triangle indices
        p_triangles = & m_topology->getTriangles();
    }
    else
    {
        p_triangles = &my_triangles;
        my_triangles.resize(newsize);
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
            my_triangles[index] = idx;
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
            my_triangles[index][0] = idx[1];
            my_triangles[index][1] = idx[2];
            my_triangles[index][2] = idx[0];
            ++index;
            my_triangles[index][0] = idx[3];
            my_triangles[index][1] = idx[0];
            my_triangles[index][2] = idx[2];
            ++index;
        }
    }
    updateFlags();
    updateNormals();
}

template<class DataTypes>
void TTriangleModel<DataTypes>::updateFlags(int /*ntri*/)
{
#if 0
    if (ntri < 0) ntri = p_triangles->size();
    //VecCoord& x =m_mstate->read(core::ConstVecCoordId::position())->getValue();
    //VecDeriv& v = m_mstate->read(core::ConstVecDerivId::velocity())->getValue();
    vector<bool> pflags(m_mstate->getSize());
    std::set<std::pair<int,int> > eflags;
    for (unsigned i=0; i<p_triangles->size(); i++)
    {
        int f = 0;
        topology::Triangle t = (*p_triangles)[i];
        if (!pflags[t[0]])
        {
            f |= FLAG_P1;
            pflags[t[0]] = true;
        }
        if (!pflags[t[1]])
        {
            f |= FLAG_P2;
            pflags[t[1]] = true;
        }
        if (!pflags[t[2]])
        {
            f |= FLAG_P3;
            pflags[t[2]] = true;
        }
        if (eflags.insert( (t[0]<t[1])?std::make_pair(t[0],t[1]):std::make_pair(t[1],t[0]) ).second)
        {
            f |= FLAG_E12;
        }
        if (i < (unsigned)ntri && eflags.insert( (t[1]<t[2])?std::make_pair(t[1],t[2]):std::make_pair(t[2],t[1]) ).second) // don't use the diagonal edge of quads
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (t[2]<t[0])?std::make_pair(t[2],t[0]):std::make_pair(t[0],t[2]) ).second)
        {
            f |= FLAG_E31;
        }
        elems[i].flags = f;
    }
#endif
}

template<class DataTypes>
void TTriangleModel<DataTypes>::handleTopologyChange()
{

    // We use the same triangle array as the topology -> only resize and recompute flags
    std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin=m_topology->beginChange();
    std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd=m_topology->endChange();
    //elems.handleTopologyEvents(itBegin,itEnd);

    while( itBegin != itEnd )
    {
        core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {


        case core::topology::ENDING_EVENT:
        {
            updateFromTopology();

            msg_info() << "TriangleModel: now " << m_topology->getNbTriangles() << " triangles.";
            resize(m_topology->getNbTriangles());
            needsUpdate=true;
            updateFlags();            
            break;
        }
        /*
        case core::topology::TRIANGLESADDED:
        {
            const sofa::component::topology::TrianglesAdded *ta=static_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
            for (unsigned int i=0;i<ta->getNbAddedTriangles();++i) {
            Triangle t(this, size - ta->getNbAddedTriangles() + i);
            const defaulttype::Vector3& pt1 = t.p1();
            const defaulttype::Vector3& pt2 = t.p2();
            const defaulttype::Vector3& pt3 = t.p3();
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();
        }
        break;
        }*/
        default: break;
        }
        ++itBegin;
    }
    return;
#if 0
    sofa::core::topology::TopologyModifier* topoMod;
    this->getContext()->get(topoMod);

    if (topoMod)   // dynamic topology
    {

        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin=m_topology->beginChange();
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd=m_topology->endChange();


        while( itBegin != itEnd )
        {
            core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

            switch( changeType )
            {


            case core::topology::ENDING_EVENT:
            {
                needsUpdate=true;
                break;
            }


            case core::topology::TRIANGLESADDED:
            {
                TriangleInfo t;
                const sofa::component::topology::TrianglesAdded *ta=static_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
                for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
                {
                    my_triangles.push_back(ta->triangleArray[i]);
                }
                resize( my_triangles.size());
                needsUpdate=true;

                break;
            }

            case core::topology::TRIANGLESREMOVED:
            {
                unsigned int last;
                unsigned int ind_last;

                last= m_topology->getNbPoints() - 1;

                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const sofa::component::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

                TriangleInfo tmp;
                topology::Triangle tmp2;

                for (unsigned int i = 0; i <tab.size(); ++i)
                {

                    unsigned int ind_k = tab[i];

                    tmp = elems[ind_k];
                    elems[ind_k] = elems[last];
                    elems[last] = tmp;

                    tmp2 = my_triangles[ind_k];
                    my_triangles[ind_k] = my_triangles[last];
                    my_triangles[last] = tmp2;

                    ind_last = elems.size() - 1;

                    if(last != ind_last)
                    {

                        tmp = elems[last];
                        elems[last] = elems[ind_last];
                        elems[ind_last] = tmp;

                        tmp2 = my_triangles[last];
                        my_triangles[last] = my_triangles[ind_last];
                        my_triangles[ind_last] = tmp2;
                    }

                    my_triangles.resize( elems.size() - 1 );
                    resize( elems.size() - 1 );

                    --last;
                }

                needsUpdate=true;

                break;
            }


            case core::topology::POINTSREMOVED:
            {
                if (m_topology->getNbTriangles()>0)
                {

                    unsigned int last = m_topology->getNbPoints() -1;

                    unsigned int i,j;
                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::helper::vector<unsigned int> lastIndexVec;
                    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
                    {

                        lastIndexVec.push_back(last - i_init);
                    }

                    for ( i = 0; i < tab.size(); ++i)
                    {
                        unsigned int i_next = i;
                        bool is_reached = false;
                        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                        {

                            i_next += 1 ;
                            is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                        }

                        if(is_reached)
                        {

                            lastIndexVec[i_next] = lastIndexVec[i];

                        }

                        const sofa::helper::vector<unsigned int> &shell=m_topology->getTrianglesAroundVertex(lastIndexVec[i]);
                        for (j=0; j<shell.size(); ++j)
                        {

                            unsigned int ind_j =shell[j];

                            if ((unsigned)my_triangles[ind_j][0]==last)
                                my_triangles[ind_j][0]=tab[i];
                            else if ((unsigned)my_triangles[ind_j][1]==last)
                                my_triangles[ind_j][1]=tab[i];
                            else if ((unsigned)my_triangles[ind_j][2]==last)
                                my_triangles[ind_j][2]=tab[i];
                        }

                        if (debug_mode)
                        {

                            for (unsigned int j_loc=0; j_loc<my_triangles.size(); ++j_loc)
                            {

                                bool is_forgotten = false;
                                if ((unsigned)my_triangles[j_loc][0]==last)
                                {
                                    my_triangles[j_loc][0]=tab[i];
                                    is_forgotten=true;

                                }
                                else
                                {
                                    if ((unsigned)my_triangles[j_loc][1]==last)
                                    {
                                        my_triangles[j_loc][1]=tab[i];
                                        is_forgotten=true;

                                    }
                                    else
                                    {
                                        if ((unsigned)my_triangles[j_loc][2]==last)
                                        {
                                            my_triangles[j_loc][2]=tab[i];
                                            is_forgotten=true;
                                        }
                                    }

                                }

                                if(is_forgotten)
                                {

                                    unsigned int ind_forgotten = j;

                                    bool is_in_shell = false;
                                    for (unsigned int j_glob=0; j_glob<shell.size(); ++j_glob)
                                    {
                                        is_in_shell = is_in_shell || (shell[j_glob] == ind_forgotten);
                                    }

                                    if(!is_in_shell)
                                    {
                                        msg_info() << "INFO_print : Col - triangle is forgotten in SHELL !!! global index = "  << ind_forgotten;
                                    }

                                }

                            }
                        }

                        --last;
                    }

                }

                needsUpdate=true;

                break;
            }

                // Case "POINTSRENUMBERING" added to propagate the treatment to the Visual Model

            case core::topology::POINTSRENUMBERING:
            {
                if (m_topology->getNbTriangles()>0)
                {

                    unsigned int i;

                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    for ( i = 0; i < my_triangles.size(); ++i)
                    {
                        my_triangles[i][0]  = tab[my_triangles[i][0]];
                        my_triangles[i][1]  = tab[my_triangles[i][1]];
                        my_triangles[i][2]  = tab[my_triangles[i][2]];
                    }

                }

                //}

                break;

            }

            default:
                // Ignore events that are not Triangle  related.
                break;
            }; // switch( changeType )

            my_triangles.resize( elems.size() ); // not necessary
            resize( elems.size() ); // not necessary

            ++itBegin;
        } // while( changeIt != last; )
    }
    if (needsUpdate)
    {
        updateFlags();
    }
#endif
}

template<class DataTypes>
void TTriangleModel<DataTypes>::draw(const core::visual::VisualParams* vparams ,int index)
{
    Element t(this,index);

    vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawTriangle( t.p1(), t.p2(), t.p3(), t.n() );
    vparams->drawTool()->setLightingEnabled(false);
}


template<class DataTypes>
void TTriangleModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        //if( size != m_topology->getNbTriangles())
        //  updateFromTopology();

        if (bothSide.getValue() || vparams->displayFlags().getShowWireFrame())
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
    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}


template<class DataTypes>
bool TTriangleModel<DataTypes>::canCollideWithElement(int index, CollisionModel* model2, int index2)
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
void TTriangleModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();

    if (needsUpdate && !cubeModel->empty()) cubeModel->resize(0);

    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    defaulttype::Vector3 minElem, maxElem;
    const VecCoord& x = this->m_mstate->read(core::ConstVecCoordId::position())->getValue();

    const bool calcNormals = computeNormals.getValue();



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
void TTriangleModel<DataTypes>::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    //updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
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
TriangleLocalMinDistanceFilter *TTriangleModel<DataTypes>::getFilter() const
{
    return m_lmdFilter;
}


template<class DataTypes>
void TTriangleModel<DataTypes>::setFilter(TriangleLocalMinDistanceFilter *lmdFilter)
{
    m_lmdFilter = lmdFilter;
}

template<class DataTypes>
int TTriangleModel<DataTypes>::getTriangleFlags(Topology::TriangleID i)
{
    int f = 0;
    sofa::core::topology::BaseMeshTopology::Triangle t = (*p_triangles)[i];

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
void TTriangleModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if( !onlyVisible ) return;

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

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}


} // namespace collision

} // namespace component

} // namespace sofa

//#endif
