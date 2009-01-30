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




void ManifoldTriangleSetTopologyModifier::createFutureModifications(sofa::helper::vector< unsigned int >& items)
{

    Triangle vertexTriangle;
    sofa::helper::vector<unsigned int> triangleVertexShell;
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;


    for (unsigned int triangleIndex = 0; triangleIndex < items.size(); triangleIndex++)
    {
        vertexTriangle = m_container->getTriangleArray()[items[triangleIndex]];

        for (unsigned int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
        {
            triangleVertexShell = m_container->getTriangleVertexShell(vertexTriangle[vertexIndex]);

            it = m_modifications.find(vertexTriangle[vertexIndex]);
            if (it == m_modifications.end())
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
            else
            {
                for (unsigned int i = 0; i < triangleVertexShell.size(); i++)
                {
                    if(triangleVertexShell[i]==items[triangleIndex])
                    {
                        m_modifications[vertexTriangle[vertexIndex]][i]=1;;
                    }
                }
            }
        }
    }
}



bool ManifoldTriangleSetTopologyModifier::testRemoveModifications()
{

    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector <PointID> border = m_container->getPointsBorder();

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
            std::cout << ((*it).second)[i] << " - " << ((*it).second)[i+1] << std::endl;

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
            m_modifications[(*it).first].erase(m_modifications[(*it).first].end()-1);
        }

        if( connexite > 2)
        {
            std::cout << "Error: Manifoldtrianglesettopologymodifier::testRemoveModifications: You could not remove this/these triangle(s)";
            std::cout << " around the vertex: " << (*it).first << std::endl;

            test=false;
        }
    }

    return test;
}




bool ManifoldTriangleSetTopologyModifier::removePrecondition(sofa::helper::vector< unsigned int >& items)
{

    createFutureModifications(items);

    return testRemoveModifications();
}


void ManifoldTriangleSetTopologyModifier::removePostProcessing()
{

    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector<unsigned int> vertexshell;


    for(it=m_modifications.begin(); it !=m_modifications.end(); it++)
    {
        std::cout << "vertex: " << (*it).first << " modifs: " << (*it).second << std::endl;
        std::cout << "shell: " << m_container->getTriangleVertexShell((*it).first)<<std::endl;

    }

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

                std::cout << vertexshell << std::endl;
                m_container->getTriangleVertexShellForModification((*it).first) = vertexshell;

                break;
            }
        }

        m_modifications.erase(it);
    }
    m_container->createEdgeVertexShellArray();

}


void ManifoldTriangleSetTopologyModifier::removePointsProcess(sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{

    for(unsigned int i = 0; i< indices.size(); i++)
    {
        m_modifications.erase(i);
    }

    TriangleSetTopologyModifier::removePointsProcess(indices,removeDOF);
}




} // namespace topology

} // namespace component

} // namespace sofa

