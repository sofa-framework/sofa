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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::init()
{
    TriangleSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
    m_mstat=dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeTetra(sofa::helper::vector<TetraID>& ind_ta)
{
    m_modifier->removeTetrahedraProcess(ind_ta,true);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideTetrahedronAlongWithPlane(TetraID ind_ta, sofa::defaulttype::Vec<3,Real>& c, sofa::defaulttype::Vec<3,Real>& normal)
{
    //Current topological state
    int nbPoint=this->m_container->getNbPoints();
    int nbTetra=this->m_container->getNbTetrahedra();

    //To be added components
    sofa::helper::vector<Vec<3,Real>>	toBeAddedPoint;
    sofa::helper::vector<int>			toBeAddedPointIndex;
    sofa::helper::vector<Tetra>			toBeAddedTetra;
    sofa::helper::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::helper::vector<TetraID>		toBeRemovedTetraIndex;
    toBeRemovedTetraIndex.push_back(ind_ta);

    //intersected edges and points
    SeqEdges intersectedEdge;
    sofa::helper::vector<Vec<3,Real>> intersectedPoint;

    //Compute intersection between tetra and plane
    const Tetrahedron ta=this->m_container->getTetra(ind_ta);
    this->m_geometryAlgorithms->getIntersectionPointWithPlane(ind_ta, c, normal,intersectedPoint,intersectedEdge);

    //For duplicating the points
    toBeAddedPointIndex.resize(intersectedPoint.size()*2);
    toBeAddedPoint.resize(intersectedPoint.size()*2);

    //added point index and coordinate
    for(int i=0; i<intersectedPoint.size(); i++)
    {
        toBeAddedPointIndex[i]=nbPoint+i;
        toBeAddedPoint[i]=intersectedPoint[i];
        //duplicated point
        toBeAddedPointIndex[i+intersectedPoint.size()]=nbPoint+i+intersectedPoint.size();
        toBeAddedPoint[i+intersectedPoint.size()]=intersectedPoint[i];
    }

    serr<<intersectedPoint.size()<<sendl;
    //Sub-division STATE1 : 3 edges are intersected by the plane
    if(intersectedEdge.size()==3)
    {
        Tetra tetra;
        if((intersectedEdge[0][0]==intersectedEdge[1][0]) || (intersectedEdge[0][0]==intersectedEdge[1][1]))
        {
            tetra[0]=intersectedEdge[0][0];
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(tetra[0]==intersectedEdge[i][j]))
                    {
                        tetra[i+1]=intersectedEdge[i][j];
                        break;
                    }
                }
            }
        }
        else
        {
            tetra[0]=intersectedEdge[0][1];
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(tetra[0]==intersectedEdge[i][j]))
                    {
                        tetra[i+1]=intersectedEdge[i][j];
                        break;
                    }
                }
            }
        }

        //generate sub-tetrahedrons
        Tetra subTetra[4];
        subTetra[0][0]=tetra[0];				subTetra[0][1]=toBeAddedPointIndex[0];	subTetra[0][2]=toBeAddedPointIndex[1];	subTetra[0][3]=toBeAddedPointIndex[2];
        subTetra[1][0]=toBeAddedPointIndex[3];	subTetra[1][1]=tetra[1];				subTetra[1][2]=tetra[2];				subTetra[1][3]=tetra[3];
        subTetra[2][0]=toBeAddedPointIndex[3];	subTetra[2][1]=toBeAddedPointIndex[4];	subTetra[2][2]=toBeAddedPointIndex[5];	subTetra[2][3]=tetra[2];
        subTetra[3][0]=toBeAddedPointIndex[3];	subTetra[3][1]=toBeAddedPointIndex[5];	subTetra[3][2]=tetra[2];				subTetra[3][3]=tetra[3];

        //add the sub tetra to list of added tetra
        for(int j=0; j<4; j++)
        {
            toBeAddedTetra.push_back(subTetra[j]);
            toBeAddedTetraIndex.push_back(nbTetra+j);
        }
    }

    //Sub-division STATE2 : 4 edges are intersected by the plane
    if(intersectedEdge.size()==4)
    {
        Tetra tetra;
        sofa::helper::vector<int> localIndex;
        localIndex.resize(4);
        tetra[0]=intersectedEdge[0][0]; tetra[1]=intersectedEdge[0][1];
        localIndex[0]=0;

        for(int j=1; j<4; j++)
        {
            while(1)
            {
                if(intersectedEdge[0][0]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][0]==intersectedEdge[j][1])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][1])
                    break;
                else
                {
                    tetra[2]=intersectedEdge[j][0]; tetra[3]=intersectedEdge[j][1];
                    break;
                }

            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==tetra[0])||(intersectedEdge[j][1]==tetra[0]))
            {
                if((intersectedEdge[j][0]==tetra[3])||(intersectedEdge[j][1]==tetra[3]))
                {
                    int temp=tetra[3];
                    tetra[3]=tetra[2];
                    tetra[2]=temp;
                }
            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==tetra[0]) || (intersectedEdge[j][1]==tetra[0]))
                localIndex[1]=j;
            else if((intersectedEdge[j][0]==tetra[1]) || (intersectedEdge[j][1]==tetra[1]))
                localIndex[3]=j;
            else
                localIndex[2]=j;
        }

        //generate sub-tetrahedrons
        Tetra subTetra[6];
        subTetra[0][0]=tetra[0];							subTetra[0][1]=toBeAddedPointIndex[localIndex[0]];	subTetra[0][2]=toBeAddedPointIndex[localIndex[1]];	subTetra[0][3]=tetra[3];
        subTetra[1][0]=toBeAddedPointIndex[localIndex[0]];	subTetra[1][1]=toBeAddedPointIndex[localIndex[1]];	subTetra[1][2]=tetra[3];							subTetra[1][3]=toBeAddedPointIndex[localIndex[3]];
        subTetra[2][0]=toBeAddedPointIndex[localIndex[1]];	subTetra[2][1]=toBeAddedPointIndex[localIndex[2]];	subTetra[2][2]=toBeAddedPointIndex[localIndex[3]];	subTetra[2][3]=tetra[3];

        subTetra[3][0]=toBeAddedPointIndex[localIndex[0]+4];	subTetra[3][1]=toBeAddedPointIndex[localIndex[1]+4];	subTetra[3][2]=toBeAddedPointIndex[localIndex[3]+4];	subTetra[3][3]=tetra[1];
        subTetra[4][0]=tetra[1];								subTetra[4][1]=toBeAddedPointIndex[localIndex[1]+4];	subTetra[4][2]=toBeAddedPointIndex[localIndex[2]+4];	subTetra[4][3]=toBeAddedPointIndex[localIndex[3]+4];
        subTetra[5][0]=tetra[1];								subTetra[5][1]=tetra[2];								subTetra[5][2]=toBeAddedPointIndex[localIndex[1]+4];	subTetra[5][3]=toBeAddedPointIndex[localIndex[2]+4];


        //add the sub tetra to list of added tetra
        for(int j=0; j<6; j++)
        {
            toBeAddedTetra.push_back(subTetra[j]);
            toBeAddedTetraIndex.push_back(nbTetra+j);
        }
    }

    //point addition (topological operation)
    m_modifier->addPointsProcess(toBeAddedPointIndex.size());
    m_modifier->addPointsWarning(toBeAddedPointIndex.size(),true);

    //point addition (geometrical operation)
    int currDof=m_mstat->getSize();
    m_mstat->resize(currDof+toBeAddedPointIndex.size());

    typename DataTypes::VecCoord& x0=*(m_mstat->getX0());
    typename DataTypes::VecCoord& x=*(m_mstat->getX());
    for(int i=0; i<toBeAddedPoint.size(); i++)
    {
        x0[currDof+i]=toBeAddedPoint[i];
        x[currDof+i]=toBeAddedPoint[i];
    }

    //tetrahedron addition and removal
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning(toBeAddedTetra.size(), (const sofa::helper::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->removeTetrahedraWarning(toBeRemovedTetraIndex);
    m_modifier->propagateTopologicalChanges();
    m_modifier->removeTetrahedraProcess(toBeRemovedTetraIndex,true);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideTetrahedronsAlongWithPlane(sofa::helper::vector<TetraID>& ind_ta, sofa::defaulttype::Vec<3,Real>& c, sofa::defaulttype::Vec<3,Real>& normal, SeqTetrahedra& subTetra)
{
    int pointIndex=this->m_container->getNbPoints();
    sofa::helper::vector<Vec<3,Real>> addedPoint;
    sofa::helper::vector<Tetra> addedTetra;
    sofa::helper::vector<Tetra> removedTetra;

    for(int i=0; i<ind_ta.size(); i++)
    {
        SeqEdges intersectedEdge;
        sofa::helper::vector<Vec<3,Real>> intersectedPoint;
        sofa::helper::vector<int> intersectedPointIndex;

        const Tetrahedron ta=this->m_container->getTetra(ind_ta[i]);
        this->m_geometryAlgorithms->getIntersectionPointWithPlane(ind_ta[i],c, normal,intersectedPoint,intersectedEdge);
        intersectedPointIndex.resize(intersectedPoint.size());

        //add the intersection points to added point list and set the intersection point index
        for(int j=0; j<intersectedPoint.size(); j++)
        {
            bool added=true;
            for(int k=0; k<addedPoint.size(); k++)
            {
                if(addedPoint[k]==intersectedPoint[j])
                {
                    intersectedPointIndex[j]=pointIndex+k;
                    added=false;
                    break;
                }
            }
            if(added)
            {
                intersectedPointIndex[j]=pointIndex+addedPoint.size();
                addedPoint.push_back(intersectedPoint[i]);
            }
        }

        //Sub-division STATE1 : 3 edges are intersected by the plane
        if(intersectedEdge.size()==3)
        {
            Tetra tetra;
            if((intersectedEdge[0][0]==intersectedEdge[1][0]) || (intersectedEdge[0][0]==intersectedEdge[1][1]))
            {
                tetra[0]=intersectedEdge[0][0];
                for(int i=0; i<3; i++)
                {
                    for(int j=0; j<3; j++)
                    {
                        if(!(tetra[0]==intersectedEdge[i][j]))
                            tetra[i]=intersectedEdge[i][j];
                    }
                }
            }
            else
            {
                tetra[0]=intersectedEdge[0][0];
                for(int i=0; i<3; i++)
                {
                    for(int j=0; j<3; j++)
                    {
                        if(!(tetra[0]==intersectedEdge[i][j]))
                            tetra[i]=intersectedEdge[i][j];
                    }
                }
            }

            //generate sub-tetrahedrons
            Tetra subTetra[4];
            subTetra[0][0]=tetra[0];					subTetra[0][1]=intersectedPointIndex[0];	subTetra[0][2]=intersectedPointIndex[1];	subTetra[0][3]=intersectedPointIndex[2];
            subTetra[1][0]=intersectedPointIndex[0];	subTetra[1][1]=tetra[1];					subTetra[1][2]=tetra[2];					subTetra[1][3]=tetra[3];
            subTetra[2][0]=intersectedPointIndex[0];	subTetra[2][1]=intersectedPointIndex[1];	subTetra[2][2]=intersectedPointIndex[2];	subTetra[2][3]=tetra[2];
            subTetra[3][0]=intersectedPointIndex[0];	subTetra[2][1]=intersectedPointIndex[2];	subTetra[2][2]=tetra[2];					subTetra[2][3]=tetra[3];

            //add the subtetra to list of added tetra
            for(int j=0; j<4; j++)
                addedTetra.push_back(subTetra[j]);

            //add the tetra to list of removed tetra
            removedTetra.push_back(tetra);
        }

        //Sub-division STATE2 : 4 edges are intersected by the plane
        if(intersectedEdge.size()==4)
        {
            Tetra tetra;
            sofa::helper::vector<int> localIndex;
            localIndex.resize(4);
            tetra[0]=intersectedEdge[0][0]; tetra[1]=intersectedEdge[0][1];
            localIndex[0]=0;

            for(int j=1; j<4; j++)
            {
                if((intersectedEdge[0][0]=!intersectedEdge[j][0]) && (intersectedEdge[0][0]=!intersectedEdge[j][1]))
                {
                    if((intersectedEdge[0][1]=!intersectedEdge[j][0]) && (intersectedEdge[0][1]=!intersectedEdge[j][1]))
                    {
                        tetra[2]=intersectedEdge[j][0]; tetra[3]=intersectedEdge[j][1];
                    }
                }
            }
            for(int j=1; j<4; j++)
            {
                if((intersectedEdge[j][0]==tetra[0])||(intersectedEdge[j][1]==tetra[0]))
                {
                    if((intersectedEdge[j][0]==tetra[3])||(intersectedEdge[j][1]==tetra[3]))
                    {
                        int temp=tetra[3];
                        tetra[3]=tetra[2];
                        tetra[2]=temp;
                    }
                }
            }
            for(int j=1; j<4; j++)
            {
                if((intersectedEdge[j][0]==tetra[0]) || (intersectedEdge[j][1]==tetra[0]))
                    localIndex[1]=j;
                else if((intersectedEdge[j][0]==tetra[1]) || (intersectedEdge[j][1]==tetra[1]))
                    localIndex[3]=j;
                else
                    localIndex[2]=j;
            }

            //generate sub-tetrahedrons
            Tetra subTetra[6];
            subTetra[0][0]=tetra[0];								subTetra[0][1]=intersectedPointIndex[localIndex[0]];	subTetra[0][2]=intersectedPointIndex[localIndex[1]];	subTetra[0][3]=tetra[3];
            subTetra[1][0]=intersectedPointIndex[localIndex[0]];	subTetra[1][1]=intersectedPointIndex[localIndex[1]];	subTetra[1][2]=tetra[3];								subTetra[1][3]=intersectedPointIndex[localIndex[3]];
            subTetra[2][0]=intersectedPointIndex[localIndex[1]];	subTetra[2][1]=intersectedPointIndex[localIndex[2]];	subTetra[2][2]=intersectedPointIndex[localIndex[3]];	subTetra[2][3]=tetra[3];

            subTetra[3][0]=intersectedPointIndex[localIndex[0]];	subTetra[3][1]=intersectedPointIndex[localIndex[1]];	subTetra[3][2]=intersectedPointIndex[localIndex[3]];	subTetra[3][3]=tetra[1];
            subTetra[4][0]=tetra[1];								subTetra[4][1]=intersectedPointIndex[localIndex[1]];	subTetra[4][2]=intersectedPointIndex[localIndex[2]];	subTetra[4][3]=intersectedPointIndex[localIndex[3]];
            subTetra[5][0]=tetra[1];								subTetra[5][1]=tetra[2];								subTetra[5][2]=intersectedPointIndex[localIndex[1]];	subTetra[5][3]=intersectedPointIndex[localIndex[2]];


            //add the subtetra to list of added tetra
            for(int j=0; j<6; j++)
                addedTetra.push_back(subTetra[j]);

            //add the tetra to list of removed tetra
            removedTetra.push_back(tetra);
        }
    }
    // Create all the points registered to be created
    //	m_modifier->addPointsProcess((const unsigned int) addedPoint.size());

    // Create all the triangles registered to be created
    //	m_modifier->addTetrahedraProcess(addedTetra) ;

    // Remove all the triangles registered to be removed
    //	m_modifier->removeTetrahedraProcess((const sofa::helper::vector< Tetra > &)removedTetra, true);
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETTOPOLOGYALGORITHMS_INL
