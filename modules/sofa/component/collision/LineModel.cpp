/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Line.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Line)

int LineModelClass = core::RegisterObject("collision model using a linear mesh, as described in MeshTopology")
        .add< LineModel >()
        .addAlias("Line")
        .addAlias("LineMeshModel")
        .addAlias("LineSetModel")
        .addAlias("LineMesh")
        .addAlias("LineSet")
        ;

//int LineSetModelClass = core::RegisterObject("collision model using a linear mesh, as described in MeshTopology")
//.add< LineSetModel >()
//;

LineModel::LineModel()
    : mstate(NULL), topology(NULL), meshRevision(-1)
{
}

//LineMeshModel::LineMeshModel()
//: meshRevision(-1), mesh(NULL)
//{
//}

//LineSetModel::LineSetModel()
//: mesh(NULL)
//{
//}

void LineModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

void LineModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    mpoints = getContext()->get<PointModel>();

    if (mstate==NULL)
    {
        std::cerr << "ERROR: LineModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (!bmt)
    {
        std::cerr << "ERROR: LineModel requires a MeshTopology.\n";
        return;
    }

    resize( bmt->getNbEdges() );

    for(int i = 0; i < bmt->getNbEdges(); i++)
    {
        elems[i].i1 = bmt->getEdge(i)[0];
        elems[i].i2 = bmt->getEdge(i)[1];
    }

    updateFromTopology();
}


void LineModel::handleTopologyChange()
{
    //if (edges != &myedges)
    //{
    // We use the same edge array as the topology -> only resize and recompute flags

    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (bmt)
    {
        resize(bmt->getNbEdges());

        for(int i = 0; i < bmt->getNbEdges(); i++)
        {
            elems[i].i1 = bmt->getEdge(i)[0];
            elems[i].i2 = bmt->getEdge(i)[1];
        }

        needsUpdate = true;
    }

    //	return;
    //}

    if (bmt)
    {
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin = bmt->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd = bmt->lastChange();

        while( itBegin != itEnd )
        {
            core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
            // Since we are using identifier, we can safely use C type casts.

            switch( changeType )
            {
            case core::componentmodel::topology::ENDING_EVENT :
            {
                //	std::cout << "INFO_print : Col - ENDING_EVENT" << std::endl;
                needsUpdate = true;
                break;
            }


            case core::componentmodel::topology::EDGESADDED :
            {
                //	std::cout << "INFO_print : Col - EDGESADDED" << std::endl;
                const sofa::component::topology::EdgesAdded *ta = dynamic_cast< const sofa::component::topology::EdgesAdded * >( *itBegin );

                for (unsigned int i = 0; i < ta->getNbAddedEdges(); ++i)
                {
                    elems[elems.size() - ta->getNbAddedEdges() + i].i1 = (ta->edgeArray[i])[0];
                    elems[elems.size() - ta->getNbAddedEdges() + i].i2 = (ta->edgeArray[i])[1];
                }

                resize( elems.size() );
                needsUpdate = true;

                break;
            }

            case core::componentmodel::topology::EDGESREMOVED :
            {
                //std::cout << "INFO_print : Col - EDGESREMOVED" << std::endl;
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

                const sofa::helper::vector< unsigned int > &tab = ( dynamic_cast< const sofa::component::topology::EdgesRemoved *>( *itBegin ) )->getArray();

                LineData tmp;
                //topology::Edge tmp2;

                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    unsigned int ind_k = tab[i];

                    tmp = elems[ind_k];
                    elems[ind_k] = elems[last];
                    elems[last] = tmp;

                    //std::cout << "INFO_print : Col - myedges.size() = " << myedges.size() << std::endl;
                    //std::cout << "INFO_print : Col - ind_k = " << ind_k << std::endl;
                    //std::cout << "INFO_print : Col - last = " << last << std::endl;

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

            case core::componentmodel::topology::POINTSREMOVED :
            {
                //std::cout << "INFO_print : Col - POINTSREMOVED" << std::endl;
                if (bmt)
                {
                    unsigned int last = bmt->getDOFNumber() - 1;

                    unsigned int i,j;
                    const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

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

                        const sofa::helper::vector<unsigned int> &shell = bmt->getEdgeVertexShell(lastIndexVec[i]);

                        for (j = 0; j < shell.size(); ++j)
                        {
                            unsigned int ind_j = shell[j];

                            if ((unsigned)elems[ind_j].i1 == last)
                            {
                                elems[ind_j].i1 = tab[i];
                            }
                            else if ((unsigned)elems[ind_j].i2 == last)
                            {
                                elems[ind_j].i2 = tab[i];
                            }
                        }

                        --last;
                    }
                }

                needsUpdate=true;

                break;
            }

            case core::componentmodel::topology::POINTSRENUMBERING:
            {
                //std::cout << "INFO_print : Vis - POINTSRENUMBERING" << std::endl;
                if (bmt)
                {
                    unsigned int i;

                    const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    for ( i = 0; i < elems.size(); ++i)
                    {
                        elems[i].i1  = tab[elems[i].i1];
                        elems[i].i2  = tab[elems[i].i2];
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

//void LineMeshModel::init()
//{
//	LineModel::init();
//	mesh = dynamic_cast< MeshTopology* > (getContext()->getTopology());
//	if (mesh==NULL)
//	{
//		std::cerr << "ERROR: LineModel requires a Mesh Topology.\n";
//		return;
//	}
//	updateFromTopology();

//	// If the CollisionDetection Method uses the filtration method based on cones
//	if (this->isFiltered())
//	{
//		// Edge neighborhood construction
//		if (mesh != NULL)
//		{
//			const int nEdges = mesh->getNbEdges();
////			if (nEdges != 0)
////			{
//				for (int i=0;i<size;i++)
//				{
//					Line l(this,i);
//					elems[i].tRight = -1;
//					elems[i].tLeft = -1;
//
//					const Vector3& pt1 = l.p1();
//					const Vector3& pt2 = l.p2();
//
//					for (int j=0; j<nEdges; j++)
//					{
//						MeshTopology::Edge idx = mesh->getEdge(j);
//						Vector3 a = (*mstate->getX())[idx[0]];
//						Vector3 b = (*mstate->getX())[idx[1]];
//						Vector3 c = (*mstate->getX())[idx[2]];
//
//						if ((a == pt1) && (b == pt2))
//							elems[i].tLeft = idx[2];
//						else if ((b == pt1) && (c == pt2))
//							elems[i].tLeft = idx[0];
//						else if ((c == pt1) && (a == pt2))
//							elems[i].tLeft = idx[1];
//						else if ((a == pt2) && (b == pt1))
//							elems[i].tRight = idx[2];
//						else if ((b == pt2) && (c == pt1))
//							elems[i].tRight = idx[0];
//						else if ((c == pt2) && (a == pt1))
//							elems[i].tRight = idx[1];
//					}
//				}
////			}
//		}
//	}
//}
///\Todo
//void LineSetModel::init()
//{
//	LineModel::init();
//	needsUpdate = true;
//// 	mesh = dynamic_cast< Topology* > (getContext()->getMainTopology());
//	sofa::simulation::tree::GNode* context = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
//	mesh = context->get< sofa::component::topology::EdgeSetTopology<DataTypes> >();
//	if (mesh==NULL)
//	{
//		std::cerr << "ERROR: LineSetModel requires a EdgeSetTopology.\n";
//		return;
//	}
//	updateFromTopology();
//	///...
//}

void LineModel::updateFromTopology()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (bmt)
    {
        int revision = bmt->getRevision();
        if (revision == meshRevision)
            return;

        needsUpdate = true;

        const unsigned int nbPoints = mstate->getX()->size();
        const unsigned int nbLines = bmt->getNbEdges();

        resize( nbLines );
        int index = 0;

        for (unsigned int i = 0; i < nbLines; i++)
        {
            topology::BaseMeshTopology::Line idx = bmt->getEdge(i);

            if (idx[0] >= nbPoints || idx[1] >= nbPoints)
            {
                std::cerr << "ERROR: Out of range index in Line " << i << ": " << idx[0] << " " << idx[1] << " ( total points = " << nbPoints << " )\n";
                continue;
            }

            elems[index].i1 = idx[0];
            elems[index].i2 = idx[1];
            ++index;
        }

        meshRevision = revision;
    }
}

//void LineMeshModel::updateFromTopology()
//{
//	needsUpdate=true;
//	int revision = mesh->getRevision();
//	if (revision == meshRevision)
//	{
//	        needsUpdate=false;
//		return;
//	}
//
//	const unsigned int npoints = mstate->getX()->size();
//	const unsigned int nlines = mesh->getNbLines();
//	resize(nlines);
//	int index = 0;
//	//VecCoord& x = *mstate->getX();
//	//VecDeriv& v = *mstate->getV();
//	for (unsigned int i=0; i<nlines; i++)
//	{
//		MeshTopology::Line idx = mesh->getLine(i);
//		if (idx[0] >= npoints || idx[1] >= npoints)
//		{
//			std::cerr << "ERROR: Out of range index in Line "<<i<<": "<<idx[0]<<" "<<idx[1]<<" ( total points="<<npoints<<")\n";
//			continue;
//		}
//		elems[index].i1 = idx[0];
//		elems[index].i2 = idx[1];
//		++index;
//	}
//	meshRevision = revision;
//	return;
//}

//void LineSetModel::updateFromTopology()
//{
//	//sofa::core::componentmodel::topology::BaseTopology* bt = mesh;
//    	sofa::component::topology::EdgeSetTopologyContainer *container = mesh->getEdgeSetTopologyContainer();
//	//needsUpdate=true;
//	if (needsUpdate)
//	{
//		const unsigned int npoints = mstate->getX()->size();
//		const unsigned int nlines = container->getNumberOfEdges();
//
//		resize(nlines);
//		int index = 0;
//		//VecCoord& x = *mstate->getX();
//		//VecDeriv& v = *mstate->getV();
//		for (unsigned int i=0; i<nlines; i++)
//		{
//			sofa::component::topology::Edge idx = container->getEdge(i);
//			if (idx[0] >= npoints || idx[1] >= npoints)
//			{
//				std::cerr << "ERROR: Out of range index in Line "<<i<<": "<<idx[0]<<" "<<idx[1]<<" ( total points="<<npoints<<")\n";
//				continue;
//			}
//
//			elems[index].i1 = idx[0];
//			elems[index].i2 = idx[1];
//			++index;
//		}
//	}
//	needsUpdate=false;
//}

void LineModel::draw(int index)
{
    Line t(this,index);
    glBegin(GL_LINES);
    helper::gl::glVertexT(t.p1());
    helper::gl::glVertexT(t.p2());
    glEnd();
}

void LineModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glColor4fv(getColor4f()); //glColor3f(0.0f, 0.0f, 1.0f);

        for (int i=0; i<size; i++) //elems.size()
        {
            if (elems[i].i1 < elems[i].i2) // only display non-edge lines
                draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
        getPrevious()->draw();
}

bool LineModel::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    if (!this->bSelfCollision.getValue()) return true;
    if (this->getContext() != model2->getContext()) return true;
    if (model2 == this)
    {
        //std::cout << "line self test "<<index<<" - "<<index2<<std::endl;
        return index < index2-2; // || index > index2+1;
    }
    else if (model2 == mpoints)
    {
        //std::cout << "line-point self test "<<index<<" - "<<index2<<std::endl;
        return index2 < elems[index].i1-1 || index2 > elems[index].i2+1;
    }
    else
        return model2->canCollideWithElement(index2, this, index);
}

void LineModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate = false;
    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Vector3 minElem, maxElem;
            Line l(this,i);
            const Vector3& pt1 = l.p1();
            const Vector3& pt2 = l.p2();

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
            }

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void LineModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Line t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3 pt1v = pt1 + t.v1()*dt;
            const Vector3 pt2v = pt2 + t.v2()*dt;

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
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

