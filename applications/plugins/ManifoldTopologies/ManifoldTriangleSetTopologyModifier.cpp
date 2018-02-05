/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "ManifoldTriangleSetTopologyModifier.h"

#include <sofa/core/visual/VisualParams.h>
#include "ManifoldTriangleSetTopologyContainer.h"
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <algorithm>
#include <iostream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
SOFA_DECL_CLASS(ManifoldTriangleSetTopologyModifier)
int ManifoldTriangleSetTopologyModifierClass = core::RegisterObject("Triangle set topology manifold modifier")
        .add< ManifoldTriangleSetTopologyModifier >()
        ;

using namespace std;
using namespace sofa::defaulttype;


void ManifoldTriangleSetTopologyModifier::init()
{
    TriangleSetTopologyModifier::init();
    this->getContext()->get(m_container);
}


void ManifoldTriangleSetTopologyModifier::reinit()
{
}


bool ManifoldTriangleSetTopologyModifier::removeTrianglesPreconditions(const sofa::helper::vector< unsigned int >& items)
{
    createRemovingTrianglesFutureModifications (items); // Create the map of modification for triangles
    createRemovingEdgesFutureModifications (items); // Create the map of modification for the edges

    return testRemovingModifications(); // Test all futures modifications
}



void ManifoldTriangleSetTopologyModifier::createRemovingTrianglesFutureModifications(const sofa::helper::vector< unsigned int >& items)
{
    Triangle vertexTriangle;
    sofa::helper::vector<unsigned int> trianglesAroundVertex;

    // Loop if there are many triangles to remove in the vector items
    for (unsigned int triangleIndex = 0; triangleIndex < items.size(); ++triangleIndex)
    {
        vertexTriangle = m_container->getTriangleArray()[items[triangleIndex]];

        // Loop on the vertex composing the triangle
        for (unsigned int vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {

            trianglesAroundVertex = m_container->getTrianglesAroundVertexForModification(vertexTriangle[vertexIndex]);

            //search in the map of modification the index of the current triangle to remove
            it_modif = m_modifications.find(vertexTriangle[vertexIndex]);

            //If not found, insert a new line in the map: key = index of triangle
            //values: vector equivalent to trianglesAroundVertex with 0 for triangles to keep and 1 for triangle to remove
            if (it_modif == m_modifications.end())
            {
                m_modifications[vertexTriangle[vertexIndex]]=sofa::helper::vector<unsigned int>();

                for (unsigned int i = 0; i < trianglesAroundVertex.size(); ++i)
                {
                    if(trianglesAroundVertex[i]==items[triangleIndex])
                    {
                        m_modifications[vertexTriangle[vertexIndex]].push_back(1);
                    }
                    else
                    {
                        m_modifications[vertexTriangle[vertexIndex]].push_back(0);
                    }
                }
            }
            else //If already exist, just change the value, of triangle to remove in the map: 0 => 1
            {
                for (unsigned int i = 0; i < trianglesAroundVertex.size(); ++i)
                {
                    if(trianglesAroundVertex[i]==items[triangleIndex])
                    {
                        m_modifications[vertexTriangle[vertexIndex]][i]=1;
                    }
                }
            }
        }
    }
}


void ManifoldTriangleSetTopologyModifier::createRemovingEdgesFutureModifications (const sofa::helper::vector <unsigned int> items)
{
    EdgesInTriangle EdgesInTriangleArray;
    bool test = true;

    for (unsigned int  i = 0; i < items.size(); ++i)
    {
        EdgesInTriangleArray = m_container->getEdgesInTriangle( items[i] );

        for (unsigned int j =0; j < 3 ; ++j)
        {

            for (unsigned int k =0; k< m_modificationsEdge.size(); ++k)
            {
                if (EdgesInTriangleArray[j] == m_modificationsEdge[k])
                {
                    test = false;
                    break;
                }
            }

            if (test)
            {
                m_modificationsEdge.push_back(EdgesInTriangleArray[j]);
            }
            test = true;
        }
    }
}



bool ManifoldTriangleSetTopologyModifier::testRemovingModifications()
{
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    const sofa::helper::vector <PointID>& border = m_container->getPointsOnBorder();

    unsigned int connexite;
    bool bord;
    bool test=true;


    for(it=m_modifications.begin(); it !=m_modifications.end(); ++it)
    {

        bord=false;

        //Test border
        for (unsigned int i = 0; i<border.size(); ++i)
        {
            if (border[i] == (*it).first)
            {
                m_modifications[(*it).first].push_back(1);
                bord=true;
            }
        }

        connexite = 0;
        for (unsigned int i = 0; i < ((*it).second).size()-1; ++i)
        {

            if( ((*it).second)[i] != ((*it).second)[i+1] )
            {
                ++connexite;
            }
        }

        //End the loop
        if( ((*it).second)[0] != ((*it).second)[((*it).second).size()-1] )
        {
            ++connexite;
        }

        if (bord)
        {
            m_modifications[(*it).first].pop_back();
        }

        if( connexite > 2)
        {
            std::cout << "Error: ManifoldTriangleSetTopologyModifier::testRemoveModifications: You could not remove this/these triangle(s)";
            std::cout << " around the vertex: " << (*it).first << std::endl;

            test=false;
        }
    }

    if( test == false )
    {
        if (!m_modifications.empty())
            m_modifications.clear();

        if (!m_modificationsEdge.empty())
            m_modificationsEdge.clear();
    }

    return test;
}




void ManifoldTriangleSetTopologyModifier::removeTrianglesPostProcessing(const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved )
{
    internalRemovingPostProcessingEdges(); // Realy apply post processings to edges of the topology.
    internalRemovingPostProcessingTriangles(); // Realy apply post processings to the triangles of the topology.

    updateRemovingModifications( edgeToBeRemoved, vertexToBeRemoved); // Update the modifications regarding isolate edges and vertex

    reorderEdgeForRemoving(); // reorder edges according to the trianglesAroundEdgeArray. Needed for edges on the "new" border.

    if (m_container->hasBorderElementLists()) // Update the list of border elements if it has been created before modifications
        m_container->createElementsOnBorder();
}



void ManifoldTriangleSetTopologyModifier::internalRemovingPostProcessingTriangles()
{
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector<unsigned int> vertexshell;

    for(it=m_modifications.begin(); it !=m_modifications.end(); ++it)
    {

        for (unsigned int i=0; i<((*it).second).size(); ++i)
        {
            if( ((*it).second)[i] == 1 )
            {
                vertexshell=m_container->getTrianglesAroundVertexForModification((*it).first);

                for (unsigned int j = 0; j<i; ++j)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getTrianglesAroundVertexForModification((*it).first) = vertexshell;

                break;
            }
        }
    }

    if (!m_modifications.empty())
        m_modifications.clear();
}



void ManifoldTriangleSetTopologyModifier::internalRemovingPostProcessingEdges()
{

    sofa::helper::vector<unsigned int> vertexshell;
    bool test = false;

    for(it_modif=m_modifications.begin(); it_modif !=m_modifications.end(); ++it_modif)
    {

        for (unsigned int i=0; i<((*it_modif).second).size(); ++i)
        {
            if( ((*it_modif).second)[i] == 1 )
            {
                test = true;
            }

            if ( ((*it_modif).second)[i] == 0 && test == true )
            {

                vertexshell=m_container->getEdgesAroundVertexForModification((*it_modif).first);

                for (unsigned int j = 0; j<i; ++j)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getEdgesAroundVertexForModification((*it_modif).first) = vertexshell;

                break;
            }
        }

        test = false;
    }
}



void ManifoldTriangleSetTopologyModifier::reorderEdgeForRemoving()
{
    for (unsigned int i = 0; i < m_modificationsEdge.size(); ++i)
    {
        reorderingEdge( m_modificationsEdge[i] );

    }

    if(!m_modificationsEdge.empty())
        m_modificationsEdge.clear();
}



void ManifoldTriangleSetTopologyModifier::updateRemovingModifications (const sofa::helper::vector< unsigned int >& edgeToBeRemoved,
        const sofa::helper::vector< unsigned int >& vertexToBeRemoved)
{
    //???
    for (unsigned int i = 0; i <vertexToBeRemoved.size(); ++i)
    {
        it_modif = m_modifications.find( vertexToBeRemoved[i] );

        if(it_modif != m_modifications.end())
            m_modifications.erase( vertexToBeRemoved[i] );
    }


    for (unsigned int i = 0; i <edgeToBeRemoved.size(); ++i)
    {
        for (unsigned int j = 0; j<m_modificationsEdge.size(); ++j)
        {
            if (m_modificationsEdge[j] == edgeToBeRemoved[i])
            {
                m_modificationsEdge.erase(m_modificationsEdge.begin()+j);
                break;
            }
        }
    }

}


void ManifoldTriangleSetTopologyModifier::Debug() //To be removed when release is sure
{
    //#ifndef NDEBUG

    std::cout << "ManifoldTriangleSetTopologyModifier::Debug()" << std::endl;

    for (int i = 0; i < m_container->getNbPoints(); ++i)
    {
        std::cout << "vertex: " << i << " => Triangles:  " << m_container->getTrianglesAroundVertexForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); ++i)
    {
        std::cout << "edge: " << i << " => Triangles:  " << m_container->getTrianglesAroundEdgeForModification(i) << std::endl;
    }

    for (int i = 0; i < m_container->getNbPoints(); ++i)
    {
        std::cout << "vertex: " << i << " => Edges:  " << m_container->getEdgesAroundVertexForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); ++i)
    {
        std::cout << "edge: " << i << " => Vertex:  " << m_container->getEdgeArray()[i] << std::endl;
    }



    //#endif
}



bool ManifoldTriangleSetTopologyModifier::addTrianglesPreconditions( const sofa::helper::vector <Triangle> &triangles)
{

    std::map< unsigned int, sofa::helper::vector <Triangle> > extremes;
    std::map< unsigned int, Triangle > trianglesList;
    std::map< unsigned int, Triangle >::iterator it;
    sofa::helper::vector <unsigned int> listDone;

    unsigned int position[3];

    bool allDone = true;
    bool oneDone = true;
    std::cout << triangles [0] << std::endl;

    // Copy the triangles vector with this positions as key:
    for (unsigned int i = 0; i < triangles.size(); i++)
    {
        trianglesList.insert ( pair <unsigned int, Triangle> (i, triangles[i]));
    }


    while ( trianglesList.size() != 0 && allDone == true)
    {
        //initialisation
        allDone = false;

        // horrible loop
        for ( it = trianglesList.begin(); it != trianglesList.end(); it++)
        {
            //	    std::cout << "it triangle: " << (*it).first << std::endl;

            oneDone = true;

            for (unsigned int vertexIndex = 0; vertexIndex <3; vertexIndex++)
            {
                //	      std::cout << "vertexIndex: " << vertexIndex << " real index: " <<(*it).second[vertexIndex] << std::endl;
                it_add = m_Addmodifications.find( (*it).second[vertexIndex]);

                //Fill map of extremes triangles and map m_addmodifications:
                if (it_add == m_Addmodifications.end())
                {
                    extremes[ (*it).second[vertexIndex] ] = sofa::helper::vector <Triangle> ();
                    m_Addmodifications[ (*it).second[vertexIndex] ] = sofa::helper::vector <int> ();

                    sofa::helper::vector <unsigned int> &trianglesAroundVertex = m_container->getTrianglesAroundVertexForModification((*it).second[vertexIndex]);

                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[trianglesAroundVertex[0]] );
                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[trianglesAroundVertex[ trianglesAroundVertex.size()-1 ]] );

                    //		std::cout << " extremes[(*it).second[vertexIndex]] " << extremes[(*it).second[vertexIndex]] << std::endl;
                    for (unsigned int i=0; i<trianglesAroundVertex.size(); i++)
                    {
                        m_Addmodifications[ (*it).second[vertexIndex] ].push_back(-1);
                    }
                }

                int vertexInTriangle0 = m_container->getVertexIndexInTriangle(extremes[(*it).second[vertexIndex]][0], (*it).second[vertexIndex]);
                int vertexInTriangle1 = m_container->getVertexIndexInTriangle(extremes[(*it).second[vertexIndex]][1], (*it).second[vertexIndex]);
                //	      std::cout << " (*it).second[ (vertexIndex+1)%3 ] " << (*it).second[ (vertexIndex+1)%3 ] << " extremes[(*it).second[vertexIndex]][1][(vertexInTriangle0+2)%3] " << extremes[(*it).second[vertexIndex]][1][(vertexInTriangle1+2)%3] << std::endl;
                //	      std::cout << " (*it).second[ (vertexIndex+2)%3 ] " << (*it).second[ (vertexIndex+2)%3 ] << " extremes[(*it).second[vertexIndex]][0][(vertexInTriangle1+1)%3] " << extremes[(*it).second[vertexIndex]][0][(vertexInTriangle0+1)%3] << std::endl;
                //Tests where the triangle could be in the shell: i.e to which extreme triangle it is adjacent ATTENTION extrems could not be similar to shell
                if ( (*it).second[ (vertexIndex+1)%3 ] == extremes[(*it).second[vertexIndex]][1][(vertexInTriangle1+2)%3] )
                {
                    //Should be added to the end of the shell
                    position[vertexIndex] = 1;
                }
                else if ( (*it).second[ (vertexIndex+2)%3 ] == extremes[(*it).second[vertexIndex]][0][(vertexInTriangle0+1)%3] )
                {
                    //Should be added to the begining of the shell
                    position[vertexIndex] = 0;
                }
                else
                {
                    oneDone = false;
                    std::cout << " Error: ManifoldTriangleSetTopologyModifier::addPrecondition adding this triangle: ";
                    std::cout << (*it).second << " doesn't keep the topology manifold." << std::endl;
                }
            }

            if (oneDone)
                //really fill m_addmodifications and update extremes map for further triangles
                for (unsigned int vertexIndex = 0; vertexIndex <3; vertexIndex++)
                {
                    extremes[ (*it).second[vertexIndex] ][ position[vertexIndex] ] = (*it).second;
                    allDone = true;

                    if ( position[vertexIndex] == 0)
                    {
                        sofa::helper::vector <int>::iterator it_vec;
                        it_vec = m_Addmodifications[ (*it).second[vertexIndex] ].begin();
                        m_Addmodifications[ (*it).second[vertexIndex] ].insert( it_vec, (int)(*it).first);
                    }
                    else if ( position[vertexIndex] == 1)
                    {
                        m_Addmodifications[ (*it).second[vertexIndex] ].push_back( (int)(*it).first );
                    }

                    listDone.push_back((*it).first); //we can't erase it while in the same loop
                }
        }

        for(unsigned int i = 0; i < listDone.size(); i++)
        {
            trianglesList.erase( listDone[i] );
        }
        listDone.clear();

        // 4 case possible:
        // - allDone = false, list.size() != 0 => never add something in the for loop, couldn't go further. leave while.
        // - allDone = true, list.zie != 0 => not all done but they were some modification, it could change in next loop for
        // - allDone = false, list.size() == 0=> normaly not possible.
        // - allDone = true , list.size() == 0 => They all have been added. Congreatulation, leave while.
    }


    //debug party:
    //	for (it_add = m_Addmodifications.begin(); it_add!=m_Addmodifications.end(); it_add++)
    //	{
    //	  std::cout << "vertex: " << (*it_add).first << " => " << (*it_add).second << std::endl;
    //	}



    if (trianglesList.size() != 0 )
    {
        //	  std::cout << " passe false" << std::endl;

        m_Addmodifications.clear();
        return false;
    }

    // For manifold classes all shells have to be present while adding triangles. As it is not obliged in upper class. It is done here.
    if(!m_container->hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle vertex shell array is empty." << std::endl;
#endif
        m_container->createTrianglesAroundVertexArray();
    }

    if(!m_container->hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle edge shell array is empty." << std::endl;
#endif
        m_container->createTrianglesAroundEdgeArray();
    }

    if(!m_container->hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Edge vertex shell array is empty." << std::endl;
#endif
        m_container->createEdgesAroundVertexArray();
    }

    //	std::cout << " passe true" << std::endl;
    return true;
}



void ManifoldTriangleSetTopologyModifier::addTrianglesPostProcessing(const sofa::helper::vector <Triangle> &triangles)
{

    // for each vertex, reorder shells:
    for (it_add = m_Addmodifications.begin(); it_add != m_Addmodifications.end(); it_add++)
    {
        //	  std::cout << " -------------------- " << std::endl;
        //	  std::cout << "it vertex: " << (*it_add).first << std::endl;
        sofa::helper::vector <unsigned int> &trianglesAroundVertex = m_container->getTrianglesAroundVertexForModification((*it_add).first);
        sofa::helper::vector <unsigned int> &edgesAroundVertex = m_container->getEdgesAroundVertexForModification((*it_add).first);

        //	  std::cout << "trianglesAroundVertex: " <<  trianglesAroundVertex << std::endl;
        //	  std::cout << "edgesAroundVertex: " <<  edgesAroundVertex << std::endl;
        sofa::helper::vector <unsigned int> triShellTmp;
        sofa::helper::vector <unsigned int> edgeShellTmp;

        triShellTmp.resize(trianglesAroundVertex.size());
        edgeShellTmp.resize(trianglesAroundVertex.size());

        bool before = true;
        unsigned int bord = 0;
//        unsigned int bord2 = 0;
        unsigned int cpt = 0;


        if ( triShellTmp.size() != ((*it_add).second).size())
            std::cout << " Error: ManifoldTriangleSetTopologyModifier::addPostProcessing: Size of shells differ. " << std::endl;

        //reordering m_trianglesAroundVertex
        for (unsigned int i = 0; i <triShellTmp.size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                triShellTmp[i] = trianglesAroundVertex[i-cpt];
            }
            else
            {
                const Triangle &tri = triangles[(*it_add).second[i]];

                int indexTriangle = m_container->getTriangleIndex(tri[0],tri[1],tri[2]);

                triShellTmp[i] = indexTriangle;
                cpt++;
            }
        }

        //	  std::cout << "triShellTmp: " << triShellTmp << std::endl;
        trianglesAroundVertex = triShellTmp;

        cpt =0;

        for (unsigned int i = 0; i <triShellTmp.size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                edgeShellTmp[i] = edgesAroundVertex[i-cpt];
                before = false;
                // bord2=i;
                bord++;
            }
            else
            {
                const Triangle &tri = triangles[(*it_add).second[i]];
                //	      std::cout << " Le tri: " << tri << std::endl;

                int vertexInTriangle = m_container->getVertexIndexInTriangle(tri, (*it_add).first);
                //	      std::cout << " le vertex dedans: " << vertexInTriangle << std::endl;


                if (before)
                {
                    //		std::cout << "passe before: "<< std::endl;
                    //		std::cout << "on cherche: " << tri[ vertexInTriangle ] << " " << tri[ (vertexInTriangle+1)%3 ]<< std::endl;

                    if ( tri[ vertexInTriangle ] < tri[ (vertexInTriangle+1)%3 ] ) // order vertex in edge
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ vertexInTriangle ], tri[ (vertexInTriangle+1)%3 ]);
                    else
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ (vertexInTriangle+1)%3 ], tri[ vertexInTriangle ]);

                    cpt++;
                    //		std::cout << "edgeShellTmp[i]: "<< edgeShellTmp[i] << std::endl;
                    //m_trianglesAroundEdge:
                    sofa::helper::vector <unsigned int> &trianglesAroundEdge = m_container->getTrianglesAroundEdgeForModification(edgeShellTmp[i]);
                    unsigned int tmp = trianglesAroundEdge[0];
                    trianglesAroundEdge[0] = trianglesAroundEdge[1];
                    trianglesAroundEdge[1] = tmp;
                }
                else
                {
                    //		std::cout << "passe after: "<< std::endl;
                    //		std::cout << "on cherche: " << tri[ vertexInTriangle ] << " " << tri[ (vertexInTriangle+1)%3 ]<< std::endl;
                    if ( tri[ vertexInTriangle ] < tri[ (vertexInTriangle+1)%3 ] )
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ vertexInTriangle ], tri[ (vertexInTriangle+1)%3 ]);
                    else
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ (vertexInTriangle+1)%3 ], tri[ vertexInTriangle ]);
                    //		std::cout << "edgeShellTmp[i]: "<< edgeShellTmp[i] << std::endl;
                    // bord2=i;
                    bord++;

                }
            }
        }
        //	  std::cout << "edgeShellTmp: " << edgeShellTmp << std::endl;

        if ( ((*it_add).second).size() != edgesAroundVertex.size()) // we are on the border, one edge is missing
        {
            //	    std::cout << "passe bord: "<< std::endl;
            //	    std::cout << "bord: " << bord << std::endl;
            //	    std::cout << "bord2: " << bord2 << std::endl;
            sofa::helper::vector <unsigned int>::iterator it;
            it = edgeShellTmp.begin();
            //	    edgeShellTmp.insert (it+bord2+1, edgesAroundVertex[bord]);
            edgeShellTmp.push_back (edgesAroundVertex[bord]);
        }

        //	  std::cout << "edgeShellTmp again: " << edgeShellTmp << std::endl;
        edgesAroundVertex = edgeShellTmp;

    }


    m_Addmodifications.clear();

    if (m_container->hasBorderElementLists()) // Update the list of border elements if it has been created before modifications
        m_container->createElementsOnBorder();
}


void ManifoldTriangleSetTopologyModifier::addRemoveTriangles (const unsigned int nTri2Add,
        const sofa::helper::vector< Triangle >& triangles2Add,
        const sofa::helper::vector< unsigned int >& trianglesIndex2Add,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        sofa::helper::vector< unsigned int >& trianglesIndex2remove)
{
    // I - Create ROI to remesh: step 1/2
    sofa::helper::vector <unsigned int> vertexROI2Remesh;

    // Look for every vertices concerned by the modifications
    for (unsigned int i = 0; i <trianglesIndex2remove.size(); i++)
    {
        Triangle new_tri = m_container->getTriangleArray()[ trianglesIndex2remove[i] ];
        for (unsigned int j = 0; j <3; j++)
        {
            vertexROI2Remesh.push_back (new_tri[j]); // these index vertex could change due to removing point.... TODO??
        }
    }


    // II - Add the triangles
    for (unsigned int i = 0; i <nTri2Add; i++)
    {
        // Use the most low level function to add triangle. I.e without any preliminary test.
        TriangleSetTopologyModifier::addTriangleProcess (triangles2Add[i]);
    }

    // Warn for the creation of all the triangles registered to be created
    TriangleSetTopologyModifier::addTrianglesWarning (nTri2Add, triangles2Add, trianglesIndex2Add, ancestors, baryCoefs);


    // III - removes the triangles

    // add the topological changes in the queue
    TriangleSetTopologyModifier::removeTrianglesWarning (trianglesIndex2remove);

    // inform other objects that the triangles are going to be removed
    propagateTopologicalChanges();

    // now destroy the old triangles.
    TriangleSetTopologyModifier::removeTrianglesProcess (trianglesIndex2remove ,true, true);


    // IV - Create ROI to remesh: step 2/2

    sofa::helper::vector <unsigned int> trianglesFinalList;
    trianglesFinalList = trianglesIndex2Add;

    std::sort( trianglesIndex2remove.begin(), trianglesIndex2remove.end(), std::greater<unsigned int>() );

    // Update the list of triangles (removing triangles change the index order)
    for (unsigned int i = 0; i<trianglesIndex2remove.size(); i++)
    {
        trianglesFinalList[trianglesFinalList.size()-1-i] = trianglesIndex2remove[i];
    }

    // Look for every vertices concerned by the modifications
    for (unsigned int i = 0; i <nTri2Add; i++)
    {
        Triangle new_tri = m_container->getTriangleArray()[ trianglesFinalList[i] ];
        for (unsigned int j = 0; j <3; j++)
        {
            vertexROI2Remesh.push_back (new_tri[j]);
        }
    }


    reorderingTopologyOnROI (vertexROI2Remesh);

    bool topo = m_container->checkTopology();
    if (!topo) //IN DEVELOPMENT (probably only in NDEBUG
    {
        std::cout <<"WARNING. [ManifoldTriangleSetTopologyModifier::addRemoveTriangles] The topology wasn't manifold after reordering the ROI. Reordering the whole triangulation." << std::endl;

        sofa::helper::vector <unsigned int> allmesh;
        for (int i = 0; i <m_container->getNbPoints(); i++)
            allmesh.push_back (i);

        reorderingTopologyOnROI (allmesh);
    }

#ifndef NDEBUG
    if(!m_container->checkTopology())
        sout << "Error. [ManifoldTriangleSetTopologyModifier::addRemoveTriangles] The topology is not any more Manifold." << endl;
#endif
}


void ManifoldTriangleSetTopologyModifier::reorderingEdge(const unsigned int edgeIndex)
{
    if(m_container->hasEdges() && m_container->hasTrianglesAroundEdge())
    {
        helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = m_container->d_edge;
        helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;

//        Edge the_edge = m_edge[edgeIndex];
        unsigned int triangleIndex, edgeIndexInTriangle;
        EdgesInTriangle EdgesInTriangleArray;
        Triangle TriangleVertexArray;

        if (m_container->m_trianglesAroundEdge[edgeIndex].empty())
        {
#ifndef NDEBUG
            std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::reorderingEdge]: shells required have not beeen created " << std::endl;
            return;
#endif
        }
        triangleIndex = m_container->m_trianglesAroundEdge[edgeIndex][0];
        EdgesInTriangleArray = m_container->getEdgesInTriangle( triangleIndex);
        TriangleVertexArray = m_triangle[triangleIndex];
        edgeIndexInTriangle = m_container->getEdgeIndexInTriangle(EdgesInTriangleArray, edgeIndex);

        m_edge[edgeIndex][0] = TriangleVertexArray[ (edgeIndexInTriangle+1)%3 ];
        m_edge[edgeIndex][1] = TriangleVertexArray[ (edgeIndexInTriangle+2)%3 ];

    }
    else
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::reorderingEdge]: shells required have not beeen created " << std::endl;
#endif
    }

}



void ManifoldTriangleSetTopologyModifier::reorderingTrianglesAroundVertex (const unsigned int vertexIndex)
{
    std::cout << "ManifoldTriangleSetTopologyModifier::reorderingTrianglesAroundVertex()" << std::endl;
    //To be added eventually
    (void)vertexIndex;
}


void ManifoldTriangleSetTopologyModifier::reorderingEdgesAroundVertex (const unsigned int vertexIndex)
{
    std::cout << "ManifoldTriangleSetTopologyModifier::reorderingEdgesAroundVertex()" << std::endl;
    //To be added eventually
    (void)vertexIndex;
}


void ManifoldTriangleSetTopologyModifier::reorderingTopologyOnROI (const sofa::helper::vector <unsigned int>& listVertex)
{
    //To use this function, all shells should have already been created.

    //Finding edges concerned
    for (unsigned int vertexIndex = 0; vertexIndex < listVertex.size(); vertexIndex++)
    {
        bool doublon = false;

        // Avoid doublon
        for(unsigned int i = 0; i<vertexIndex; i++)
        {
            if (listVertex[i] == listVertex[vertexIndex]) //means this new vertex has already been treated
            {
                doublon = true;
                break;
            }
        }

        if (doublon)
            continue;

        // Check if the vertex really exist
        if ( (int)listVertex[ vertexIndex ] >= m_container->getNbPoints())
        {
#ifndef NDEBUG
            std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::reorderingTopologyOnROI]: vertex: "<< listVertex[ vertexIndex ] << " is out of bound" << std::endl;
#endif
            continue;
        }

        // Start processing:

        sofa::helper::vector <unsigned int>& edgesAroundVertex = m_container->getEdgesAroundVertexForModification( listVertex[vertexIndex] );
        sofa::helper::vector <unsigned int>& trianglesAroundVertex = m_container->getTrianglesAroundVertexForModification( listVertex[vertexIndex] );

        sofa::helper::vector <unsigned int>::iterator it;
        sofa::helper::vector < sofa::helper::vector <unsigned int> > vertexTofind;

        sofa::helper::vector <unsigned int> goodEdgeShell;
        sofa::helper::vector <unsigned int> goodTriangleShell;

        unsigned int firstVertex =0;
        unsigned int secondVertex =0;
        unsigned int cpt = 0;

        vertexTofind.resize (trianglesAroundVertex.size());
        helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;


        // Path to follow creation
        for (unsigned int triangleIndex = 0; triangleIndex < trianglesAroundVertex.size(); triangleIndex++)
        {
            Triangle vertexTriangle = m_triangle[ trianglesAroundVertex[triangleIndex] ];

            vertexTofind[triangleIndex].push_back( vertexTriangle[ ( m_container->getVertexIndexInTriangle(vertexTriangle, listVertex[vertexIndex] )+1 )%3 ]);
            vertexTofind[triangleIndex].push_back( vertexTriangle[ ( m_container->getVertexIndexInTriangle(vertexTriangle, listVertex[vertexIndex] )+2 )%3 ]);
        }
        firstVertex = vertexTofind[0][0];
        secondVertex = vertexTofind[0][1];

        goodTriangleShell.push_back(trianglesAroundVertex[0]);

        int the_edge = m_container->getEdgeIndex ( listVertex [vertexIndex], vertexTofind[0][0]);
        if (the_edge == -1)
            the_edge = m_container->getEdgeIndex ( vertexTofind[0][0], listVertex [vertexIndex]);
        goodEdgeShell.push_back(the_edge);

        bool testFind = false;
        bool reverse = false;
        cpt = 0;

        // Start following path
        for (unsigned int triangleIndex = 1; triangleIndex < trianglesAroundVertex.size(); triangleIndex++)
        {
            for (unsigned int pathIndex = 1; pathIndex < trianglesAroundVertex.size(); pathIndex++)
            {

                if (vertexTofind[pathIndex][0] == secondVertex)
                {
                    goodTriangleShell.push_back(trianglesAroundVertex[pathIndex]);

                    int the_edge = m_container->getEdgeIndex ( listVertex [vertexIndex], vertexTofind[pathIndex][0]);
                    if (the_edge == -1)
                        the_edge = m_container->getEdgeIndex ( vertexTofind[pathIndex][0], listVertex [vertexIndex]);
                    goodEdgeShell.push_back(the_edge);

                    secondVertex = vertexTofind[pathIndex][1];

                    testFind = true;
                    break;
                }
            }

            if (!testFind) //tetra has not be found, this mean we reach a border, we reverse the method
            {
                reverse = true;
                break;
            }

            cpt++;
            testFind = false;
        }

        // Reverse path following methode
        if(reverse)
        {
#ifndef NDEBUG
            std::cout << "shell on border: "<< listVertex[vertexIndex] << std::endl;
#endif

            for (unsigned int triangleIndex = cpt+1; triangleIndex<trianglesAroundVertex.size(); triangleIndex++)
            {
                for (unsigned int pathIndex = 0; pathIndex<trianglesAroundVertex.size(); pathIndex++)
                {

                    if (vertexTofind[pathIndex][1] == firstVertex)
                    {
                        goodTriangleShell.insert (goodTriangleShell.begin(),trianglesAroundVertex[pathIndex]);

                        int the_edge = m_container->getEdgeIndex ( listVertex [vertexIndex], vertexTofind[pathIndex][0]);
                        if (the_edge == -1)
                            the_edge = m_container->getEdgeIndex ( vertexTofind[pathIndex][1], listVertex [vertexIndex]);
                        goodEdgeShell.insert (goodEdgeShell.begin(),the_edge);

                        firstVertex = vertexTofind[pathIndex][0];
                        break;
                    }
                }
            }
        }

        for (unsigned int i = 0; i<vertexTofind.size(); i++)
        {
            for (unsigned int j = vertexIndex; j <listVertex.size(); j++)
            {
                if (vertexTofind[i][0] == listVertex[j])
                {
                    int the_edge = m_container->getEdgeIndex ( listVertex [vertexIndex], listVertex [j]);

                    if (the_edge == -1)
                        the_edge = m_container->getEdgeIndex ( listVertex [j], listVertex [vertexIndex]);

                    reorderingEdge (the_edge);
                }
            }
        }

        if (edgesAroundVertex.size() != trianglesAroundVertex.size()) //border case
        {
            bool edgeFind = false;
            int the_edge = -1;

            for (unsigned int i = 0; i < edgesAroundVertex.size(); i++)
            {
                edgeFind = false;
                for (unsigned int j = 0; j < goodEdgeShell.size(); j++)
                {
                    if (edgesAroundVertex[i] == goodEdgeShell[j])
                    {
                        edgeFind = true;
                        break;
                    }
                }

                if(!edgeFind)
                {
                    the_edge = edgesAroundVertex[i];
                    break;
                }
            }

            if (the_edge != -1)
            {
                goodEdgeShell.push_back(the_edge);
                reorderingEdge(the_edge);
            }
            else
            {
#ifndef NDEBUG
                std::cout << "Error: reorderingTopologyOnROI: vertex "<< listVertex[vertexIndex] << "is on the border but last edge not found." <<std::endl;
#endif
            }
        }

        edgesAroundVertex = goodEdgeShell;
        goodEdgeShell.clear();
        vertexTofind.clear();

        trianglesAroundVertex = goodTriangleShell;
        goodTriangleShell.clear();
    }
}




} // namespace topology

} // namespace component

} // namespace sofa

