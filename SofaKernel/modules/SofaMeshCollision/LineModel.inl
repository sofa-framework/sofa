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

#ifndef SOFA_COMPONENT_COLLISION_LINEMODEL_INL
#define SOFA_COMPONENT_COLLISION_LINEMODEL_INL

#include <SofaMeshCollision/LineModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaMeshCollision/LineLocalMinDistanceFilter.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/Line.h>
#include <sofa/core/CollisionElement.h>
#include <vector>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using core::topology::BaseMeshTopology;

template<class DataTypes>
TLineModel<DataTypes>::TLineModel()
    : bothSide(initData(&bothSide, false, "bothSide", "activate collision on both side of the line model (when surface normals are defined on these lines)") )
    , mstate(NULL), topology(NULL), meshRevision(-1), m_lmdFilter(NULL)
    , LineActiverPath(initData(&LineActiverPath,"LineActiverPath", "path of a component LineActiver that activates or deactivates collision line during execution") )
    , m_displayFreePosition(initData(&m_displayFreePosition, false, "displayFreePosition", "Display Collision Model Points free position(in green)") )
{
    enum_type = LINE_TYPE;
}

//LineMeshModel::LineMeshModel()
//: meshRevision(-1), mesh(NULL)
//{
//}

//LineSetModel::LineSetModel()
//: mesh(NULL)
//{
//}

template<class DataTypes>
void TLineModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

template<class DataTypes>
void TLineModel<DataTypes>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
//    mpoints = getContext()->get<sofa::component::collision::PointModel>();
    this->getContext()->get(mpoints);

    if (mstate==NULL)
    {
        serr << "LineModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    simulation::Node* node = dynamic_cast< simulation::Node* >(this->getContext());
    if (node != 0)
    {
        m_lmdFilter = node->getNodeObject< LineLocalMinDistanceFilter >();
    }

    core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (!bmt)
    {
        serr <<"LineModel requires a MeshTopology" << sendl;
        return;
    }
	this->topology = bmt;
    resize( bmt->getNbEdges() );

    for(int i = 0; i < bmt->getNbEdges(); i++)
    {
        elems[i].p[0] = bmt->getEdge(i)[0];
        elems[i].p[1] = bmt->getEdge(i)[1];
    }

    updateFromTopology();

    const std::string path = LineActiverPath.getValue();

    if (path.size()==0)
    {

        myActiver = LineActiver::getDefaultActiver();
        sout<<"path = "<<path<<" no Line Activer found for LineModel "<<this->getName()<<sendl;
    }
    else
    {

        core::objectmodel::BaseObject *activer=NULL;
        this->getContext()->get(activer ,path  );

        if (activer != NULL)
            sout<<" Activer named"<<activer->getName()<<" found"<<sendl;
        else
            serr<<"wrong path for Line Activer"<<sendl;


        myActiver = dynamic_cast<LineActiver *> (activer);



        if (myActiver==NULL)
        {
            myActiver = LineActiver::getDefaultActiver();


            serr<<"wrong path for Line Activer for LineModel "<< this->getName() <<sendl;
        }
        else
        {
            sout<<"Line Activer named"<<activer->getName()<<" found !! for LineModel "<< this->getName() <<sendl;
        }
    }

}

template<class DataTypes>
void TLineModel<DataTypes>::handleTopologyChange()
{
    //if (edges != &myedges)
    //{
    // We use the same edge array as the topology -> only resize and recompute flags

    core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (bmt)
    {
        resize(bmt->getNbEdges());

        for(int i = 0; i < bmt->getNbEdges(); i++)
        {
            elems[i].p[0] = bmt->getEdge(i)[0];
            elems[i].p[1] = bmt->getEdge(i)[1];
        }

        needsUpdate = true;
    }

    //	return;
    //}

    if (bmt)
    {
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin = bmt->beginChange();
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd = bmt->endChange();

        while( itBegin != itEnd )
        {
            core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

            switch( changeType )
            {
            case core::topology::ENDING_EVENT :
            {
                //	sout << "INFO_print : Col - ENDING_EVENT" << sendl;
                needsUpdate = true;
                break;
            }


            case core::topology::EDGESADDED :
            {
                //	sout << "INFO_print : Col - EDGESADDED" << sendl;
                const core::topology::EdgesAdded *ta = static_cast< const core::topology::EdgesAdded * >( *itBegin );

                for (unsigned int i = 0; i < ta->getNbAddedEdges(); ++i)
                {
                    elems[elems.size() - ta->getNbAddedEdges() + i].p[0] = (ta->edgeArray[i])[0];
                    elems[elems.size() - ta->getNbAddedEdges() + i].p[1] = (ta->edgeArray[i])[1];
                }

                resize( elems.size() );
                needsUpdate = true;

                break;
            }

            case core::topology::EDGESREMOVED :
            {
                //sout << "INFO_print : Col - EDGESREMOVED" << sendl;
                unsigned int last;
                unsigned int ind_last;

                if (bmt)
                {
                    last = bmt->getNbEdges() - 1;
                }
                else
                {
                    last = elems.size() -1;
                }

                const sofa::helper::vector< unsigned int > &tab = ( static_cast< const core::topology::EdgesRemoved *>( *itBegin ) )->getArray();

                LineData tmp;
                //topology::Edge tmp2;

                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    unsigned int ind_k = tab[i];

                    tmp = elems[ind_k];
                    elems[ind_k] = elems[last];
                    elems[last] = tmp;

                    //sout << "INFO_print : Col - myedges.size() = " << myedges.size() << sendl;
                    //sout << "INFO_print : Col - ind_k = " << ind_k << sendl;
                    //sout << "INFO_print : Col - last = " << last << sendl;

                    //tmp2 = myedges[ind_k];
                    //myedges[ind_k] = myedges[last];
                    //myedges[last] = tmp2;

                    ind_last = elems.size() - 1;

                    if(last != ind_last)
                    {
                        tmp = elems[last];
                        elems[last] = elems[ind_last];
                        elems[ind_last] = tmp;

                        //tmp2 = myedges[last];
                        //myedges[last] = myedges[ind_last];
                        //myedges[ind_last] = tmp2;
                    }

                    //myedges.resize( elems.size() - 1 );
                    resize( elems.size() - 1 );

                    --last;
                }

                needsUpdate=true;

                break;
            }

            case core::topology::POINTSREMOVED :
            {
                //sout << "INFO_print : Col - POINTSREMOVED" << sendl;
                if (bmt)
                {
                    unsigned int last = bmt->getNbPoints() - 1;

                    unsigned int i,j;
                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const core::topology::PointsRemoved * >( *itBegin ) )->getArray();

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

                        const sofa::helper::vector<unsigned int> &shell = bmt->getEdgesAroundVertex(lastIndexVec[i]);

                        for (j = 0; j < shell.size(); ++j)
                        {
                            unsigned int ind_j = shell[j];

                            if ((unsigned)elems[ind_j].p[0] == last)
                            {
                                elems[ind_j].p[0] = tab[i];
                            }
                            else if ((unsigned)elems[ind_j].p[1] == last)
                            {
                                elems[ind_j].p[1] = tab[i];
                            }
                        }

                        --last;
                    }
                }

                needsUpdate=true;

                break;
            }

            case core::topology::POINTSRENUMBERING:
            {
                //sout << "INFO_print : Vis - POINTSRENUMBERING" << sendl;
                if (bmt)
                {
                    unsigned int i;

                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    for ( i = 0; i < elems.size(); ++i)
                    {
                        elems[i].p[0]  = tab[elems[i].p[0]];
                        elems[i].p[1]  = tab[elems[i].p[1]];
                    }
                }

                break;
            }

            default:
                // Ignore events that are not Edge  related.
                break;
            }; // switch( changeType )

            resize( elems.size() ); // not necessary

            ++itBegin;
        } // while( changeIt != last; )
    }
}

template<class DataTypes>
void TLineModel<DataTypes>::updateFromTopology()
{
    core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (bmt)
    {
        int revision = bmt->getRevision();
        if (revision == meshRevision)
            return;

        needsUpdate = true;

        const unsigned int nbPoints = mstate->getSize();
        const unsigned int nbLines = bmt->getNbEdges();

        resize( nbLines );
        int index = 0;

        for (unsigned int i = 0; i < nbLines; i++)
        {
            core::topology::BaseMeshTopology::Line idx = bmt->getEdge(i);

            if (idx[0] >= nbPoints || idx[1] >= nbPoints)
            {
                serr << "ERROR: Out of range index in Line " << i << ": " << idx[0] << " " << idx[1] << " : total points (size of the MState) = " << nbPoints <<sendl;
                continue;
            }

            elems[index].p[0] = idx[0];
            elems[index].p[1] = idx[1];
            ++index;
        }

        meshRevision = revision;
    }
}

template<class DataTypes>
void TLineModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,true);

        std::vector< defaulttype::Vector3 > points;
        for (int i=0; i<size; i++)
        {
            TLine<DataTypes> l(this,i);
            if(l.activated())
            {
                points.push_back(l.p1());
                points.push_back(l.p2());
            }
        }

        vparams->drawTool()->drawLines(points, 1, defaulttype::Vec<4,float>(getColor4f()));

        if (m_displayFreePosition.getValue())
        {
            std::vector< defaulttype::Vector3 > pointsFree;
            for (int i=0; i<size; i++)
            {
                TLine<DataTypes> l(this,i);
                if(l.activated())
                {
                    pointsFree.push_back(l.p1Free());
                    pointsFree.push_back(l.p2Free());
                }
            }

            vparams->drawTool()->drawLines(pointsFree, 1, defaulttype::Vec<4,float>(0.0f,1.0f,0.2f,1.0f));
        }

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,false);
    }
    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}

template<class DataTypes>
bool TLineModel<DataTypes>::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    //msg_info()<<"canCollideWithElement is called"<<std::endl;

    if (!this->bSelfCollision.getValue()) return true;
    if (this->getContext() != model2->getContext()) return true;
    sofa::core::topology::BaseMeshTopology* topology = this->getMeshTopology();
    /*
    	TODO : separate 2 case: the model is only composed of lines or is composed of triangles
    	bool NoTriangles = true;
    	if( this->getContext()->get<TriangleModel>  != NULL)
    		NoTriangles = false;
    */
    int p11 = elems[index].p[0];
    int p12 = elems[index].p[1];


    if (!topology)
    {
        serr<<"no topology found"<<sendl;
        return true;
    }
    const helper::vector <unsigned int>& EdgesAroundVertex11 =topology->getEdgesAroundVertex(p11);
    const helper::vector <unsigned int>& EdgesAroundVertex12 =topology->getEdgesAroundVertex(p12);
    //msg_info()<<"EdgesAroundVertex11 ok"<<std::endl;


    if (model2 == this)
    {
        // if point in common, return false:
        int p21 = elems[index2].p[0];
        int p22 = elems[index2].p[1];

        if (p11==p21 || p11==p22 || p12==p21 || p12==p22)
            return false;


        // in the neighborhood, if we find a segment in common, we cancel the collision
        const helper::vector <unsigned int>& EdgesAroundVertex21 =topology->getEdgesAroundVertex(p21);
        const helper::vector <unsigned int>& EdgesAroundVertex22 =topology->getEdgesAroundVertex(p22);

        for (unsigned int i1=0; i1<EdgesAroundVertex11.size(); i1++)
        {
            unsigned int e11 = EdgesAroundVertex11[i1];
            unsigned int i2;
            for (i2=0; i2<EdgesAroundVertex21.size(); i2++)
            {
                if (e11==EdgesAroundVertex21[i2])
                    return false;
            }
            for (i2=0; i2<EdgesAroundVertex22.size(); i2++)
            {
                if (e11==EdgesAroundVertex22[i2])
                    return false;
            }
        }

        for (unsigned int i1=0; i1<EdgesAroundVertex12.size(); i1++)
        {
            unsigned int e11 = EdgesAroundVertex12[i1];
            unsigned int i2;
            for (i2=0; i2<EdgesAroundVertex21.size(); i2++)
            {
                if (e11==EdgesAroundVertex21[i2])
                    return false;
            }
            for (i2=0; i2<EdgesAroundVertex22.size(); i2++)
            {
                if (e11==EdgesAroundVertex22[i2])
                    return false;
            }

        }
        return true;



    }
    else if (model2 == mpoints)
    {
        //msg_info()<<" point Model"<<std::endl;

        // if point belong to the segment, return false
        if (index2==p11 || index2==p12)
            return false;

        // if the point belong to the a segment in the neighborhood, return false
        for (unsigned int i1=0; i1<EdgesAroundVertex11.size(); i1++)
        {
            unsigned int e11 = EdgesAroundVertex11[i1];
            p11 = elems[e11].p[0];
            p12 = elems[e11].p[1];
            if (index2==p11 || index2==p12)
                return false;
        }
        for (unsigned int i1=0; i1<EdgesAroundVertex12.size(); i1++)
        {
            unsigned int e12 = EdgesAroundVertex12[i1];
            p11 = elems[e12].p[0];
            p12 = elems[e12].p[1];
            if (index2==p11 || index2==p12)
                return false;
        }
        return true;

        //sout << "line-point self test "<<index<<" - "<<index2<<sendl;
        //std::cout << "line-point self test "<<index<<" - "<<index2<<"   - elems[index].p[0]-1"<<elems[index].p[0]-1<<"   elems[index]..p[1]+1 "<<elems[index].p[1]+1<<std::endl;


        // case1: only lines (aligned lines !!)
        //return index2 < p11-1 || index2 > p12+1;

        // only removes collision with the two vertices of the segment
        // TODO: neighborhood search !
        //return !(index2==p11 || index2==p12);
    }


    else
        return model2->canCollideWithElement(index2, this, index);
}

template<class DataTypes>
void TLineModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate = false;
    defaulttype::Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            defaulttype::Vector3 minElem, maxElem;
            TLine<DataTypes> l(this,i);
            const defaulttype::Vector3& pt1 = l.p1();
            const defaulttype::Vector3& pt2 = l.p2();

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }

    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}

template<class DataTypes>
void TLineModel<DataTypes>::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
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
            TLine<DataTypes> t(this,i);
            const defaulttype::Vector3& pt1 = t.p1();
            const defaulttype::Vector3& pt2 = t.p2();
            const defaulttype::Vector3 pt1v = pt1 + t.v1()*dt;
            const defaulttype::Vector3 pt2v = pt2 + t.v2()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
int TLineModel<DataTypes>::getLineFlags(int i)
{
    int f = 0;
    if (topology)
    {
        sofa::core::topology::BaseMeshTopology::Edge e(elems[i].p[0], elems[i].p[1]);
        i = getElemEdgeIndex(i);
        if ((unsigned)i < (unsigned)topology->getNbEdges())
        {
            for (unsigned int j=0; j<2; ++j)
            {
                const sofa::core::topology::BaseMeshTopology::EdgesAroundVertex& eav = topology->getEdgesAroundVertex(e[j]);
                if (eav[0] == (sofa::core::topology::BaseMeshTopology::EdgeID)i)
                    f |= (FLAG_P1 << j);
                if (eav.size() == 1)
                    f |= (FLAG_BP1 << j);
            }
        }
    }
    return f;
}

template<class DataTypes>
LineLocalMinDistanceFilter *TLineModel<DataTypes>::getFilter() const
{
    return m_lmdFilter;
}

template<class DataTypes>
void TLineModel<DataTypes>::setFilter(LineLocalMinDistanceFilter *lmdFilter)
{
    m_lmdFilter = lmdFilter;
}

template<class DataTypes>
void TLineModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if( !onlyVisible ) return;

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::min();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (int i=0; i<size; i++)
    {
        Element e(this,i);
        const defaulttype::Vector3& pt1 = e.p1();
        const defaulttype::Vector3& pt2 = e.p2();

        for (int c=0; c<3; c++)
        {
            if (pt1[c] > maxBBox[c]) maxBBox[c] = (Real)pt1[c];
            else if (pt1[c] < minBBox[c]) minBBox[c] = (Real)pt1[c];

            if (pt2[c] > maxBBox[c]) maxBBox[c] = (Real)pt2[c];
            else if (pt2[c] < minBBox[c]) minBBox[c] = (Real)pt2[c];
        }
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
