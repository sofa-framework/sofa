/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/topology/ManifoldTriangleSetTopologyModifier.h>
//#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <algorithm>
//#include <functional>
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
    if(!(m_triSwap.getValue()).empty() && this->getContext()->getAnimate()) //temporarly test for the funciton edgeSwap
    {
        edgeSwapProcess (m_triSwap.getValue());
    }

    if(m_swapMesh.getValue() && this->getContext()->getAnimate())
    {
        swapRemeshing();
    }

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
    sofa::helper::vector<unsigned int> triangleVertexShell;

    // Loop if there are many triangles to remove in the vector items
    for (unsigned int triangleIndex = 0; triangleIndex < items.size(); triangleIndex++)
    {
        vertexTriangle = m_container->getTriangleArray()[items[triangleIndex]];

        // Loop on the vertex composing the triangle
        for (unsigned int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
        {

            triangleVertexShell = m_container->getTriangleVertexShellForModification(vertexTriangle[vertexIndex]);

            //search in the map of modification the index of the current triangle to remove
            it_modif = m_modifications.find(vertexTriangle[vertexIndex]);

            //If not found, insert a new line in the map: key = index of triangle
            //values: vector equivalent to triangleVertexShell with 0 for triangles to keep and 1 for triangle to remove
            if (it_modif == m_modifications.end())
            {
                m_modifications[vertexTriangle[vertexIndex]]=sofa::helper::vector<unsigned int>();

                for (unsigned int i = 0; i < triangleVertexShell.size(); i++)
                {
                    if(triangleVertexShell[i]==items[triangleIndex])
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
                for (unsigned int i = 0; i < triangleVertexShell.size(); i++)
                {
                    if(triangleVertexShell[i]==items[triangleIndex])
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
    TriangleEdges TriangleEdgeArray;
    bool test = true;

    for (unsigned int  i = 0; i < items.size(); i++)
    {
        TriangleEdgeArray = m_container->getTriangleEdge( items[i] );

        for (unsigned int j =0; j < 3 ; j++)
        {

            for (unsigned int k =0; k< m_modificationsEdge.size(); k++)
            {
                if (TriangleEdgeArray[j] == m_modificationsEdge[k])
                {
                    test = false;
                    break;
                }
            }

            if (test)
            {
                m_modificationsEdge.push_back(TriangleEdgeArray[j]);
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


    for(it=m_modifications.begin(); it !=m_modifications.end(); it++)
    {

        bord=false;

        //Test border
        for (unsigned int i = 0; i<border.size(); i++)
        {
            if (border[i] == (*it).first)
            {
                m_modifications[(*it).first].push_back(1);
                bord=true;
            }
        }

        connexite = 0;
        for (unsigned int i = 0; i < ((*it).second).size()-1; i++)
        {

            if( ((*it).second)[i] != ((*it).second)[i+1] )
            {
                connexite++;
            }
        }

        //End the loop
        if( ((*it).second)[0] != ((*it).second)[((*it).second).size()-1] )
        {
            connexite++;
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

    reorderEdgeForRemoving(); // reorder edges according to the triangleEdgeShellArray. Needed for edges on the "new" border.

    if (m_container->hasBorderElementLists()) // Update the list of border elements if it has been created before modifications
        m_container->createElementsOnBorder();
}



void ManifoldTriangleSetTopologyModifier::internalRemovingPostProcessingTriangles()
{
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector<unsigned int> vertexshell;

    for(it=m_modifications.begin(); it !=m_modifications.end(); ++it)
    {

        for (unsigned int i=0; i<((*it).second).size(); i++)
        {
            if( ((*it).second)[i] == 1 )
            {
                vertexshell=m_container->getTriangleVertexShellForModification((*it).first);

                for (unsigned int j = 0; j<i; j++)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getTriangleVertexShellForModification((*it).first) = vertexshell;

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

        for (unsigned int i=0; i<((*it_modif).second).size(); i++)
        {
            if( ((*it_modif).second)[i] == 1 )
            {
                test = true;
            }

            if ( ((*it_modif).second)[i] == 0 && test == true )
            {

                vertexshell=m_container->getEdgeVertexShellForModification((*it_modif).first);

                for (unsigned int j = 0; j<i; j++)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getEdgeVertexShellForModification((*it_modif).first) = vertexshell;

                break;
            }
        }

        test = false;
    }
}



void ManifoldTriangleSetTopologyModifier::reorderEdgeForRemoving()
{
    for (unsigned int i = 0; i < m_modificationsEdge.size(); i++)
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
    for (unsigned int i = 0; i <vertexToBeRemoved.size(); i++)
    {
        it_modif = m_modifications.find( vertexToBeRemoved[i] );

        if(it_modif != m_modifications.end())
            m_modifications.erase( vertexToBeRemoved[i] );
    }


    for (unsigned int i = 0; i <edgeToBeRemoved.size(); i++)
    {
        for (unsigned int j = 0; j<m_modificationsEdge.size(); j++)
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

    for (int i = 0; i < m_container->getNbPoints(); i++)
    {
        std::cout << "vertex: " << i << " => Triangles:  " << m_container->getTriangleVertexShellForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); i++)
    {
        std::cout << "edge: " << i << " => Triangles:  " << m_container->getTriangleEdgeShellForModification(i) << std::endl;
    }

    for (int i = 0; i < m_container->getNbPoints(); i++)
    {
        std::cout << "vertex: " << i << " => Edges:  " << m_container->getEdgeVertexShellForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); i++)
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

                    sofa::helper::vector <unsigned int> &triangleVertexShell = m_container->getTriangleVertexShellForModification((*it).second[vertexIndex]);

                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[triangleVertexShell[0]] );
                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[triangleVertexShell[ triangleVertexShell.size()-1 ]] );

                    //		std::cout << " extremes[(*it).second[vertexIndex]] " << extremes[(*it).second[vertexIndex]] << std::endl;
                    for (unsigned int i=0; i<triangleVertexShell.size(); i++)
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
    if(!m_container->hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle vertex shell array is empty." << std::endl;
#endif
        m_container->createTriangleVertexShellArray();
    }

    if(!m_container->hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle edge shell array is empty." << std::endl;
#endif
        m_container->createTriangleEdgeShellArray();
    }

    if(!m_container->hasEdgeVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Edge vertex shell array is empty." << std::endl;
#endif
        m_container->createEdgeVertexShellArray();
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
        sofa::helper::vector <unsigned int> &triangleVertexShell = m_container->getTriangleVertexShellForModification((*it_add).first);
        sofa::helper::vector <unsigned int> &edgeVertexShell = m_container->getEdgeVertexShellForModification((*it_add).first);

        //	  std::cout << "triangleVertexShell: " <<  triangleVertexShell << std::endl;
        //	  std::cout << "edgeVertexShell: " <<  edgeVertexShell << std::endl;
        sofa::helper::vector <unsigned int> triShellTmp;
        sofa::helper::vector <unsigned int> edgeShellTmp;

        triShellTmp.resize(triangleVertexShell.size());
        edgeShellTmp.resize(triangleVertexShell.size());

        bool before = true;
        unsigned int bord = 0;
        unsigned int bord2 = 0;
        unsigned int cpt = 0;


        if ( triShellTmp.size() != ((*it_add).second).size())
            std::cout << " Error: ManifoldTriangleSetTopologyModifier::addPostProcessing: Size of shells differ. " << std::endl;

        //reordering m_triangleVertexShell
        for (unsigned int i = 0; i <triShellTmp.size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                triShellTmp[i] = triangleVertexShell[i-cpt];
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
        triangleVertexShell = triShellTmp;

        cpt =0;

        for (unsigned int i = 0; i <triShellTmp.size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                edgeShellTmp[i] = edgeVertexShell[i-cpt];
                before = false;
                bord2=i;
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
                    //m_triangleEdgeShell:
                    sofa::helper::vector <unsigned int> &triangleEdgeShell = m_container->getTriangleEdgeShellForModification(edgeShellTmp[i]);
                    unsigned int tmp = triangleEdgeShell[0];
                    triangleEdgeShell[0] = triangleEdgeShell[1];
                    triangleEdgeShell[1] = tmp;
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
                    bord2=i;
                    bord++;

                }
            }
        }
        //	  std::cout << "edgeShellTmp: " << edgeShellTmp << std::endl;

        if ( ((*it_add).second).size() != edgeVertexShell.size()) // we are on the border, one edge is missing
        {
            //	    std::cout << "passe bord: "<< std::endl;
            //	    std::cout << "bord: " << bord << std::endl;
            //	    std::cout << "bord2: " << bord2 << std::endl;
            sofa::helper::vector <unsigned int>::iterator it;
            it = edgeShellTmp.begin();
            //	    edgeShellTmp.insert (it+bord2+1, edgeVertexShell[bord]);
            edgeShellTmp.push_back (edgeVertexShell[bord]);
        }

        //	  std::cout << "edgeShellTmp again: " << edgeShellTmp << std::endl;
        edgeVertexShell = edgeShellTmp;

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
        TriangleSetTopologyModifier::addSingleTriangleProcess (triangles2Add[i]);
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






void ManifoldTriangleSetTopologyModifier::edgeSwapProcess (const sofa::helper::vector <EdgeID>& listEdges)
{

    for (unsigned int i = 0; i<listEdges.size(); i++)
    {
        edgeSwap(listEdges[i]);
        propagateTopologicalChanges();
    }
}


void ManifoldTriangleSetTopologyModifier::edgeSwapProcess (const TriangleID& indexTri1, const TriangleID& indexTri2)
{
    sofa::helper::vector < unsigned int > listVertex;
    unsigned int cpt = 0;
    int commonEdgeIndex;
    bool test = true;
    Edge commonEdge;

    Triangle vertexTriangle1 = m_container->getTriangleArray()[indexTri1];
    for (unsigned int i = 0; i < 3; i++)
        listVertex.push_back(vertexTriangle1[i]);

    Triangle vertexTriangle2 = m_container->getTriangleArray()[indexTri2];
    for (unsigned int i = 0; i <3; i++)
    {
        test =true;
        for (unsigned int j = 0; j <3; j++)
        {
            if (vertexTriangle2[i] == listVertex[j])
            {
                commonEdge[cpt] = vertexTriangle2[i];
                cpt++;
                test = false;
                break;
            }
        }

        if (test)
            listVertex.push_back(vertexTriangle2[i]);
    }


    if (commonEdge[0] < commonEdge[1])
        commonEdgeIndex = m_container->getEdgeIndex(commonEdge[0], commonEdge[1]);
    else
        commonEdgeIndex = m_container->getEdgeIndex(commonEdge[1], commonEdge[0]);

    if (commonEdgeIndex == -1 || listVertex.size() > 4)
    {
        std::cout << "Error: edgeSwapProcess: the two selected triangles are not adjacent" << std::endl;
        return;
    }
    else
    {
        edgeSwap(commonEdgeIndex);
        propagateTopologicalChanges();
    }
}



void ManifoldTriangleSetTopologyModifier::edgeSwap(const EdgeID& edgeIndex)
{

    sofa::helper::vector < unsigned int > listVertex;
    sofa::helper::vector< Triangle > triToAdd; triToAdd.resize (2);
    sofa::helper::vector< TriangleID > triToAddID; triToAddID.resize (2);
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors; ancestors.resize(2);
    sofa::helper::vector< sofa::helper::vector< double > > baryCoefs; baryCoefs.resize (2);
    sofa::helper::vector< TriangleID > trianglesIndex2remove; trianglesIndex2remove.resize(2);

    trianglesIndex2remove = m_container->getTriangleEdgeShellArray()[edgeIndex];

    if(trianglesIndex2remove.size()>2)
    {
        std::cout << "Error: edgeSwap: the topology is not manifold around the input edge: "<< edgeIndex << std::endl;
        return;
    }
    else if (trianglesIndex2remove.size() == 1)
    {
        std::cout << "Error: edgeSwap: the edge: "<< edgeIndex << " is on the border of the mesh. Swaping this edge is impossible" << std::endl;
        return;
    }

    int edgeInTri1 = m_container->getEdgeIndexInTriangle ( m_container->getEdgeTriangleShell (trianglesIndex2remove[0]), edgeIndex);
    int edgeInTri2 = m_container->getEdgeIndexInTriangle ( m_container->getEdgeTriangleShell (trianglesIndex2remove[1]), edgeIndex);
    Triangle vertexTriangle1 = m_container->getTriangle (trianglesIndex2remove[0]);
    Triangle vertexTriangle2 = m_container->getTriangle (trianglesIndex2remove[1]);

    Triangle newTri;

    newTri[0] = vertexTriangle1[ edgeInTri1 ];
    newTri[1] = vertexTriangle1[ (edgeInTri1+1)%3 ];
    newTri[2] = vertexTriangle2[ edgeInTri2 ];
    triToAdd[0] = newTri;

    listVertex.push_back (newTri[0]);
    listVertex.push_back (newTri[1]);
    listVertex.push_back (newTri[2]);

    newTri[0] = vertexTriangle2[ edgeInTri2 ];
    newTri[1] = vertexTriangle2[ (edgeInTri2+1)%3 ];
    newTri[2] = vertexTriangle1[ edgeInTri1 ];
    triToAdd[1] = newTri;

    listVertex.push_back (newTri[1]);

    for (unsigned int i = 0; i <2; i++)
    {
        ancestors[i].push_back (trianglesIndex2remove[0]); baryCoefs[i].push_back (0.5);
        ancestors[i].push_back (trianglesIndex2remove[1]); baryCoefs[i].push_back (0.5);
    }
    triToAddID[0] = m_container->getNbTriangles();
    triToAddID[1] = m_container->getNbTriangles()+1;

    addRemoveTriangles (triToAdd.size(), triToAdd, triToAddID, ancestors, baryCoefs, trianglesIndex2remove);
}


void ManifoldTriangleSetTopologyModifier::reorderingEdge(const unsigned int edgeIndex)
{

    if(m_container->hasEdges() && m_container->hasTriangleEdgeShell())
    {
        Edge the_edge = m_container->m_edge[edgeIndex];
        unsigned int triangleIndex, edgeIndexInTriangle;
        TriangleEdges TriangleEdgeArray;
        Triangle TriangleVertexArray;

        if (m_container->m_triangleEdgeShell[edgeIndex].empty())
        {
#ifndef NDEBUG
            std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::reorderingEdge]: shells required have not beeen created " << std::endl;
            return;
#endif
        }

        triangleIndex = m_container->m_triangleEdgeShell[edgeIndex][0];
        TriangleEdgeArray = m_container->getTriangleEdge( triangleIndex);
        TriangleVertexArray = m_container->m_triangle[triangleIndex];

        edgeIndexInTriangle = m_container->getEdgeIndexInTriangle(TriangleEdgeArray, edgeIndex);

        m_container->m_edge[edgeIndex][0] = TriangleVertexArray[ (edgeIndexInTriangle+1)%3 ];
        m_container->m_edge[edgeIndex][1] = TriangleVertexArray[ (edgeIndexInTriangle+2)%3 ];

    }
    else
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::reorderingEdge]: shells required have not beeen created " << std::endl;
#endif
    }

}


void ManifoldTriangleSetTopologyModifier::reorderingTriangleVertexShell (const unsigned int vertexIndex)
{
    std::cout << "ManifoldTriangleSetTopologyModifier::reorderingTriangleVertexShell()" << std::endl;
    //To be added eventually
    (void)vertexIndex;
}


void ManifoldTriangleSetTopologyModifier::reorderingEdgeVertexShell (const unsigned int vertexIndex)
{
    std::cout << "ManifoldTriangleSetTopologyModifier::reorderingEdgeVertexShell()" << std::endl;
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

        sofa::helper::vector <unsigned int>& edgeVertexShell = m_container->getEdgeVertexShellForModification( listVertex[vertexIndex] );
        sofa::helper::vector <unsigned int>& triangleVertexShell = m_container->getTriangleVertexShellForModification( listVertex[vertexIndex] );

        sofa::helper::vector <unsigned int>::iterator it;
        sofa::helper::vector < sofa::helper::vector <unsigned int> > vertexTofind;

        sofa::helper::vector <unsigned int> goodEdgeShell;
        sofa::helper::vector <unsigned int> goodTriangleShell;

        unsigned int firstVertex =0;
        unsigned int secondVertex =0;
        unsigned int cpt = 0;

        vertexTofind.resize (triangleVertexShell.size());

        // Path to follow creation
        for (unsigned int triangleIndex = 0; triangleIndex < triangleVertexShell.size(); triangleIndex++)
        {
            Triangle vertexTriangle = m_container->m_triangle[ triangleVertexShell[triangleIndex] ];

            vertexTofind[triangleIndex].push_back( vertexTriangle[ ( m_container->getVertexIndexInTriangle(vertexTriangle, listVertex[vertexIndex] )+1 )%3 ]);
            vertexTofind[triangleIndex].push_back( vertexTriangle[ ( m_container->getVertexIndexInTriangle(vertexTriangle, listVertex[vertexIndex] )+2 )%3 ]);
        }
        firstVertex = vertexTofind[0][0];
        secondVertex = vertexTofind[0][1];

        goodTriangleShell.push_back(triangleVertexShell[0]);

        int the_edge = m_container->getEdgeIndex ( listVertex [vertexIndex], vertexTofind[0][0]);
        if (the_edge == -1)
            the_edge = m_container->getEdgeIndex ( vertexTofind[0][0], listVertex [vertexIndex]);
        goodEdgeShell.push_back(the_edge);

        bool testFind = false;
        bool reverse = false;
        cpt = 0;

        // Start following path
        for (unsigned int triangleIndex = 1; triangleIndex < triangleVertexShell.size(); triangleIndex++)
        {
            for (unsigned int pathIndex = 1; pathIndex < triangleVertexShell.size(); pathIndex++)
            {

                if (vertexTofind[pathIndex][0] == secondVertex)
                {
                    goodTriangleShell.push_back(triangleVertexShell[pathIndex]);

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

            for (unsigned int triangleIndex = cpt+1; triangleIndex<triangleVertexShell.size(); triangleIndex++)
            {
                for (unsigned int pathIndex = 0; pathIndex<triangleVertexShell.size(); pathIndex++)
                {

                    if (vertexTofind[pathIndex][1] == firstVertex)
                    {
                        goodTriangleShell.insert (goodTriangleShell.begin(),triangleVertexShell[pathIndex]);

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

        if (edgeVertexShell.size() != triangleVertexShell.size()) //border case
        {
            bool edgeFind = false;
            int the_edge = -1;

            for (unsigned int i = 0; i < edgeVertexShell.size(); i++)
            {
                edgeFind = false;
                for (unsigned int j = 0; j < goodEdgeShell.size(); j++)
                {
                    if (edgeVertexShell[i] == goodEdgeShell[j])
                    {
                        edgeFind = true;
                        break;
                    }
                }

                if(!edgeFind)
                {
                    the_edge = edgeVertexShell[i];
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

        edgeVertexShell = goodEdgeShell;
        goodEdgeShell.clear();
        vertexTofind.clear();

        triangleVertexShell = goodTriangleShell;
        goodTriangleShell.clear();
    }
}


void ManifoldTriangleSetTopologyModifier::swapRemeshing()
{
    // All the mesh is about to be remeshed by swaping edges. So passing a simple list.
    sofa::helper::vector <EdgeID> listEdges;
    for(unsigned int i = 0; i<m_container->getNumberOfEdges(); i++)
        listEdges.push_back (i);

    swapRemeshing(listEdges);
}


void ManifoldTriangleSetTopologyModifier::swapRemeshing(sofa::helper::vector <EdgeID>& listEdges)
{
    //sofa::helper::vector <EdgeID> edgeToSwap;
    bool allDone = false;

    while (!allDone)
    {
        allDone = true;
        for (unsigned int edgeIndex = 0; edgeIndex<listEdges.size() ; edgeIndex++)
        {
            const sofa::helper::vector <TriangleID>& shell = m_container->getTriangleEdgeShellArray()[listEdges[edgeIndex]];

            if (shell.size() == 2)
            {
                sofa::helper::vector <unsigned int> listVertex;
                const sofa::helper::vector <PointID>& border = m_container->getPointsOnBorder();
                TriangleID indexTri1, indexTri2;

                indexTri1 = shell[0];
                indexTri2 = shell[1];

                int edgeInTri1 = m_container->getEdgeIndexInTriangle ( m_container->getEdgeTriangleShell (indexTri1), listEdges[edgeIndex]);
                int edgeInTri2 = m_container->getEdgeIndexInTriangle ( m_container->getEdgeTriangleShell (indexTri2), listEdges[edgeIndex]);
                Triangle vertexTriangle1 = m_container->getTriangleArray()[indexTri1];
                Triangle vertexTriangle2 = m_container->getTriangleArray()[indexTri2];

                listVertex.push_back( vertexTriangle1[edgeInTri1] );
                listVertex.push_back( vertexTriangle2[edgeInTri2] );
                listVertex.push_back( vertexTriangle1[ (edgeInTri1+1)%3 ] );
                listVertex.push_back( vertexTriangle2[ (edgeInTri2+1)%3 ] );

                int sum = 0;

                sum = (m_container->getTriangleVertexShellArray()[ listVertex[0] ]).size();
                sum += (m_container->getTriangleVertexShellArray()[ listVertex[1] ]).size();
                sum -= (m_container->getTriangleVertexShellArray()[ listVertex[2] ]).size();
                sum -= (m_container->getTriangleVertexShellArray()[ listVertex[3] ]).size();

                for (unsigned int i = 0; i <2; i++)
                {
                    for (unsigned int j = 0; j <border.size(); j++)
                    {
                        if(listVertex[i] == border[j])
                        {
                            sum+=2;
                            break;
                        }
                    }
                }

                for (unsigned int i = 2; i <4; i++)
                {
                    for (unsigned int j = 0; j <border.size(); j++)
                    {
                        if(listVertex[i] == border[j])
                        {
                            sum-=2;
                            break;
                        }
                    }
                }

                if (sum < -2)
                {
                    //edgeToSwap.push_back (listEdges[edgeIndex]);
                    edgeSwap (listEdges[edgeIndex]);
                    propagateTopologicalChanges();
                    allDone = false;
                }
            }
        }

        //edgeSwapProcess (edgeToSwap);
        //edgeToSwap.clear();
    }
}



} // namespace topology

} // namespace component

} // namespace sofa

