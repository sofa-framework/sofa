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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <SofaBaseTopology/TetrahedronSetTopologyAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::init()
{
    TriangleSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
    m_intialNbPoints=m_container->getNbPoints();
    m_baryLimit=0.2f;
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeTetra(sofa::helper::vector<TetraID>& ind_ta)
{
    m_modifier->removeTetrahedraProcess(ind_ta,true);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideTetrahedronsWithPlane(sofa::helper::vector< sofa::helper::vector<double> >& coefs, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    int nbPoint=this->m_container->getNbPoints();
    int nbTetra=this->m_container->getNbTetrahedra();

    //Number of to be added points
    unsigned int nbTobeAddedPoints=(unsigned int)intersectedEdgeID.size()*2;

    //barycentric coodinates of to be added points
    sofa::helper::vector< sofa::helper::vector<unsigned int> > ancestors;
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::helper::vector< unsigned int > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);
        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::helper::vector<Tetra>			toBeAddedTetra;
    sofa::helper::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::helper::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::helper::vector<TetraID> intersectedTetras;
    sofa::helper::vector<sofa::helper::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( unsigned int j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( unsigned int k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( unsigned int i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::helper::vector<unsigned int> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( unsigned int j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( unsigned int k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=nbPoint+k*2;
                }
            }
        }
        nbTobeAddedTetras+=subDivideTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning((unsigned int)toBeAddedTetra.size(), (const sofa::helper::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    sout << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size() << sendl;
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideTetrahedronsWithPlane(sofa::helper::vector<Coord>& intersectedPoints, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    int nbPoint=this->m_container->getNbPoints();
    int nbTetra=this->m_container->getNbTetrahedra();

    //Number of to be added points
    unsigned int nbTobeAddedPoints=(unsigned int)intersectedEdgeID.size()*2;

    //barycentric coodinates of to be added points
    sofa::helper::vector< sofa::helper::vector<unsigned int> > ancestors;
    sofa::helper::vector< sofa::helper::vector<double> > coefs;
    for( unsigned int i=0; i<intersectedPoints.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::defaulttype::Vec<3,double> p;
        p[0]=intersectedPoints[i][0]; p[1]=intersectedPoints[i][1]; p[2]=intersectedPoints[i][2];
        sofa::helper::vector< double > coef = m_geometryAlgorithms->compute2PointsBarycoefs(p, theEdge[0], theEdge[1]);

        sofa::helper::vector< unsigned int > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);

        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
        coefs.push_back(coef); coefs.push_back(coef);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::helper::vector<Tetra>			toBeAddedTetra;
    sofa::helper::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::helper::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::helper::vector<TetraID> intersectedTetras;
    sofa::helper::vector<sofa::helper::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( unsigned int j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( unsigned int k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( unsigned int i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::helper::vector<unsigned int> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( unsigned int j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( unsigned int k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=nbPoint+k*2;
                }
            }
        }
        nbTobeAddedTetras+=subDivideTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning((unsigned int)toBeAddedTetra.size(), (const sofa::helper::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    sout << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size() << sendl;
}

template<class DataTypes>
int TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideTetrahedronWithPlane(TetraID tetraIdx, sofa::helper::vector<EdgeID>& intersectedEdgeID, sofa::helper::vector<unsigned int>& intersectedPointID, Coord planeNormal, sofa::helper::vector<Tetra>& toBeAddedTetra)
{
    Tetra intersectedTetra=this->m_container->getTetra(tetraIdx);
    int nbAddedTetra;
    Coord edgeDirec;

    //1. Number of intersected edge = 1
    if(intersectedEdgeID.size()==1)
    {
        Edge intersectedEdge=this->m_container->getEdge(intersectedEdgeID[0]);
        sofa::helper::vector<unsigned int> pointsID;

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int j=0; j<4; j++)
        {
            if(!(intersectedTetra[j]==intersectedEdge[0]))
            {
                if(!(intersectedTetra[j]==intersectedEdge[1]))
                    pointsID.push_back(intersectedTetra[j]);
            }
        }

        //construct subdivided tetrahedrons
        Tetra subTetra[2];
        edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;
        Real dot=edgeDirec*planeNormal;

        //inspect the tetrahedron is already subdivided
        if((pointsID[0]>=m_intialNbPoints) && (pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[0]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[1]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0]+1;
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0];
            }
            else
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0];
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0]+1;
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<2; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=2;
        return nbAddedTetra;
    }

    //2. Number of intersected edge = 2
    if(intersectedEdgeID.size()==2)
    {
        Edge intersectedEdge[2];
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        //find the point index which included both intersected edge
        if(intersectedEdge[0][0]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][0]==intersectedEdge[1][1])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][1]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0]);
        }
        else
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0]);
        }

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int i=0; i<4; i++)
        {
            bool flag=true;
            for(int j=0; j<3; j++)
            {
                if(intersectedTetra[i]==pointsID[j])
                {
                    flag=false;
                    break;
                }
            }
            if(flag)
            {
                pointsID[3]=intersectedTetra[i];
                break;
            }
        }

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[3];

        if(pointsID[3]>=m_intialNbPoints)
        {
            if((pointsID[3]-m_intialNbPoints)%2==0)//normal �ݴ�
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3]+1;
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]+1;
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]-1;
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]-1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3]-1;
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<3; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=3;
        return nbAddedTetra;
    }

    //3. Number of intersected edge = 3
    if(intersectedEdgeID.size()==3)
    {
        int DIVISION_STATE=0;			//1: COMPLETE DIVISION, 2: PARTIAL DIVISION
        Edge intersectedEdge[3];

        //sorting
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        if(intersectedEdgeID[1]>intersectedEdgeID[2])
        {
            int temp=intersectedEdgeID[1];
            intersectedEdgeID[1]=intersectedEdgeID[2];
            intersectedEdgeID[2]=temp;

            temp=intersectedPointID[1];
            intersectedPointID[1]=intersectedPointID[2];
            intersectedPointID[2]=temp;
        }
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }

        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        for(int i=1; i<3; i++)
        {
            if(intersectedEdge[0][0]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][0]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
        }

        //determine devision state
        int nbEdgeSharingPoint=0;
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<2; j++)
            {
                if(pointsID[0]==intersectedEdge[i][j])
                    nbEdgeSharingPoint++;
            }
        }
        if(nbEdgeSharingPoint==3)
            DIVISION_STATE=1;
        if(nbEdgeSharingPoint==2)
            DIVISION_STATE=2;

        //DIVISION STATE 1
        if(DIVISION_STATE==1)
        {
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0]);
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(pointsID[0]==intersectedEdge[i][j]))
                        pointsID[i+1]=intersectedEdge[i][j];
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[4];
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=intersectedPointID[2]+1;
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=intersectedPointID[2];	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=intersectedPointID[2];	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2];	subTetra[3][1]=pointsID[1];				subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=intersectedPointID[2];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1;	subTetra[1][2]=intersectedPointID[2]+1;	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=intersectedPointID[2]+1;	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2]+1;	subTetra[3][1]=pointsID[1]			;	subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<4; i++)
            {
                if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
                {
                    unsigned int temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=4;
            return nbAddedTetra;
        }

        //DIVISION STATE 2
        if(DIVISION_STATE==2)
        {
            Coord edgeDirec;
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0]);

            int secondIntersectedEdgeIndex = 0, thirdIntersectedEdgeIndex = 0;
            int conectedIndex/*, nonConectedIndex*/;

            if(pointsID[0]==intersectedEdge[0][0])
                pointsID[1]=intersectedEdge[0][1];
            else
                pointsID[1]=intersectedEdge[0][0];

            if(pointsID[0]==intersectedEdge[1][0])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[1][1])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][0])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][1])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[3]==intersectedEdge[thirdIntersectedEdgeIndex][0])
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][1])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }
            else
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][0])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[5];

            if(dot>0)
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];				subTetra[2][1]=pointsID[3];				subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];				subTetra[3][1]=intersectedPointID[0];	subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];				subTetra[4][1]=pointsID[2];				subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
            }
            else
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<5; i++)
            {
                if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
                {
                    unsigned int temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=5;
            return nbAddedTetra;
            return 0;
        }
    }

    //Sub-division STATE2 : 4 edges are intersected by the plane
    if(intersectedEdgeID.size()==4)
    {
        Edge intersectedEdge[4];
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);
        intersectedEdge[3]=this->m_container->getEdge(intersectedEdgeID[3]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        sofa::helper::vector<unsigned int> localIndex;
        localIndex.resize(4);
        localIndex[0]=0;

        pointsID[0]=intersectedEdge[0][0]; pointsID[1]=intersectedEdge[0][1];
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
                    pointsID[2]=intersectedEdge[j][0]; pointsID[3]=intersectedEdge[j][1];
                    break;
                }

            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0])||(intersectedEdge[j][1]==pointsID[0]))
            {
                if((intersectedEdge[j][0]==pointsID[3])||(intersectedEdge[j][1]==pointsID[3]))
                {
                    int temp=pointsID[3];
                    pointsID[3]=pointsID[2];
                    pointsID[2]=temp;
                }
            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0]) || (intersectedEdge[j][1]==pointsID[0]))
                localIndex[1]=j;
            else if((intersectedEdge[j][0]==pointsID[1]) || (intersectedEdge[j][1]==pointsID[1]))
                localIndex[3]=j;
            else
                localIndex[2]=j;
        }

        Coord edgeDirec;
        edgeDirec=this->m_geometryAlgorithms->computeEdgeDirection(intersectedEdgeID[0])*-1;

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[6];
        if(dot>0)
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]]+1;	subTetra[0][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[2]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[2]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]];	subTetra[3][2]=intersectedPointID[localIndex[2]];	subTetra[3][3]=intersectedPointID[localIndex[3]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
        }
        else
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[2]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]];	subTetra[0][3]=intersectedPointID[localIndex[3]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]]+1;	subTetra[3][2]=intersectedPointID[localIndex[2]]+1;	subTetra[3][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
        }

        for(int i=0; i<6; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=6;
        return nbAddedTetra;
    }
    return 0;
}








































template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideRestTetrahedronsWithPlane(sofa::helper::vector< sofa::helper::vector<double> >& coefs, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    int nbPoint=this->m_container->getNbPoints();
    int nbTetra=this->m_container->getNbTetrahedra();

    //Number of to be added points
    unsigned int nbTobeAddedPoints=(unsigned int)intersectedEdgeID.size()*2;

    //barycentric coodinates of to be added points
    sofa::helper::vector< sofa::helper::vector<unsigned int> > ancestors;
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::helper::vector< unsigned int > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);
        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::helper::vector<Tetra>			toBeAddedTetra;
    sofa::helper::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::helper::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::helper::vector<TetraID> intersectedTetras;
    sofa::helper::vector<sofa::helper::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( unsigned int j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( unsigned int k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( unsigned int i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::helper::vector<unsigned int> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( unsigned int j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( unsigned int k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=nbPoint+k*2;
                }
            }
        }
        nbTobeAddedTetras+=subDivideRestTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning((unsigned int)toBeAddedTetra.size(), (const sofa::helper::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    sout << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size() << sendl;
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideRestTetrahedronsWithPlane(sofa::helper::vector<Coord>& intersectedPoints, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    int nbPoint=this->m_container->getNbPoints();
    int nbTetra=this->m_container->getNbTetrahedra();

    //Number of to be added points
    unsigned int nbTobeAddedPoints=(unsigned int)intersectedEdgeID.size()*2;

    //barycentric coodinates of to be added points
    sofa::helper::vector< sofa::helper::vector<unsigned int> > ancestors;
    sofa::helper::vector< sofa::helper::vector<double> > coefs;
    for( unsigned int i=0; i<intersectedPoints.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::defaulttype::Vec<3,double> p;
        p[0]=intersectedPoints[i][0]; p[1]=intersectedPoints[i][1]; p[2]=intersectedPoints[i][2];
        sofa::helper::vector< double > coef = m_geometryAlgorithms->computeRest2PointsBarycoefs(p, theEdge[0], theEdge[1]);

        sofa::helper::vector< unsigned int > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);

        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
        coefs.push_back(coef); coefs.push_back(coef);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::helper::vector<Tetra>			toBeAddedTetra;
    sofa::helper::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::helper::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::helper::vector<TetraID> intersectedTetras;
    sofa::helper::vector<sofa::helper::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( unsigned int i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( unsigned int j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( unsigned int k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( unsigned int i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::helper::vector<unsigned int> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( unsigned int j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( unsigned int k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=nbPoint+k*2;
                }
            }
        }
        nbTobeAddedTetras+=subDivideRestTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning((unsigned int)toBeAddedTetra.size(), (const sofa::helper::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    sout << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size() << sendl;
}

template<class DataTypes>
int TetrahedronSetTopologyAlgorithms< DataTypes >::subDivideRestTetrahedronWithPlane(TetraID tetraIdx, sofa::helper::vector<EdgeID>& intersectedEdgeID, sofa::helper::vector<unsigned int>& intersectedPointID, Coord planeNormal, sofa::helper::vector<Tetra>& toBeAddedTetra)
{
    Tetra intersectedTetra=this->m_container->getTetra(tetraIdx);
    int nbAddedTetra;
    Coord edgeDirec;

    //1. Number of intersected edge = 1
    if(intersectedEdgeID.size()==1)
    {
        Edge intersectedEdge=this->m_container->getEdge(intersectedEdgeID[0]);
        sofa::helper::vector<unsigned int> pointsID;

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int j=0; j<4; j++)
        {
            if(!(intersectedTetra[j]==intersectedEdge[0]))
            {
                if(!(intersectedTetra[j]==intersectedEdge[1]))
                    pointsID.push_back(intersectedTetra[j]);
            }
        }

        //construct subdivided tetrahedrons
        Tetra subTetra[2];
        edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        Real dot=edgeDirec*planeNormal;

        //inspect the tetrahedron is already subdivided
        if((pointsID[0]>=m_intialNbPoints) && (pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[0]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[1]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0]+1;
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0];
            }
            else
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0];
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0]+1;
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<2; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=2;
        return nbAddedTetra;
    }

    //2. Number of intersected edge = 2
    if(intersectedEdgeID.size()==2)
    {
        Edge intersectedEdge[2];
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        //find the point index which included both intersected edge
        if(intersectedEdge[0][0]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][0]==intersectedEdge[1][1])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][1]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0]);
        }
        else
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0]);
        }

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int i=0; i<4; i++)
        {
            bool flag=true;
            for(int j=0; j<3; j++)
            {
                if(intersectedTetra[i]==pointsID[j])
                {
                    flag=false;
                    break;
                }
            }
            if(flag)
            {
                pointsID[3]=intersectedTetra[i];
                break;
            }
        }

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[3];

        if(pointsID[3]>=m_intialNbPoints)
        {
            if((pointsID[3]-m_intialNbPoints)%2==0)//normal �ݴ�
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3]+1;
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]+1;
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]-1;
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]-1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3]-1;
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<3; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=3;
        return nbAddedTetra;
    }

    //3. Number of intersected edge = 3
    if(intersectedEdgeID.size()==3)
    {
        int DIVISION_STATE=0;			//1: COMPLETE DIVISION, 2: PARTIAL DIVISION
        Edge intersectedEdge[3];

        //sorting
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        if(intersectedEdgeID[1]>intersectedEdgeID[2])
        {
            int temp=intersectedEdgeID[1];
            intersectedEdgeID[1]=intersectedEdgeID[2];
            intersectedEdgeID[2]=temp;

            temp=intersectedPointID[1];
            intersectedPointID[1]=intersectedPointID[2];
            intersectedPointID[2]=temp;
        }
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }

        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        for(int i=1; i<3; i++)
        {
            if(intersectedEdge[0][0]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][0]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
        }

        //determine devision state
        int nbEdgeSharingPoint=0;
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<2; j++)
            {
                if(pointsID[0]==intersectedEdge[i][j])
                    nbEdgeSharingPoint++;
            }
        }
        if(nbEdgeSharingPoint==3)
            DIVISION_STATE=1;
        if(nbEdgeSharingPoint==2)
            DIVISION_STATE=2;

        //DIVISION STATE 1
        if(DIVISION_STATE==1)
        {
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0]);
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(pointsID[0]==intersectedEdge[i][j]))
                        pointsID[i+1]=intersectedEdge[i][j];
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[4];
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=intersectedPointID[2]+1;
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=intersectedPointID[2];	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=intersectedPointID[2];	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2];	subTetra[3][1]=pointsID[1];				subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=intersectedPointID[2];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1;	subTetra[1][2]=intersectedPointID[2]+1;	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=intersectedPointID[2]+1;	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2]+1;	subTetra[3][1]=pointsID[1]			;	subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<4; i++)
            {
                if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
                {
                    unsigned int temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=4;
            return nbAddedTetra;
        }

        //DIVISION STATE 2
        if(DIVISION_STATE==2)
        {
            Coord edgeDirec;
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0]);

            int secondIntersectedEdgeIndex = 0, thirdIntersectedEdgeIndex = 0;
            int conectedIndex/*, nonConectedIndex*/;

            if(pointsID[0]==intersectedEdge[0][0])
                pointsID[1]=intersectedEdge[0][1];
            else
                pointsID[1]=intersectedEdge[0][0];

            if(pointsID[0]==intersectedEdge[1][0])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[1][1])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][0])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][1])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[3]==intersectedEdge[thirdIntersectedEdgeIndex][0])
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][1])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }
            else
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][0])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[5];

            if(dot>0)
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];				subTetra[2][1]=pointsID[3];				subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];				subTetra[3][1]=intersectedPointID[0];	subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];				subTetra[4][1]=pointsID[2];				subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
            }
            else
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<5; i++)
            {
                if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
                {
                    unsigned int temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=5;
            return nbAddedTetra;
            return 0;
        }
    }

    //Sub-division STATE2 : 4 edges are intersected by the plane
    if(intersectedEdgeID.size()==4)
    {
        Edge intersectedEdge[4];
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);
        intersectedEdge[3]=this->m_container->getEdge(intersectedEdgeID[3]);

        sofa::helper::vector<unsigned int> pointsID;
        pointsID.resize(4);

        sofa::helper::vector<unsigned int> localIndex;
        localIndex.resize(4);
        localIndex[0]=0;

        pointsID[0]=intersectedEdge[0][0]; pointsID[1]=intersectedEdge[0][1];
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
                    pointsID[2]=intersectedEdge[j][0]; pointsID[3]=intersectedEdge[j][1];
                    break;
                }

            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0])||(intersectedEdge[j][1]==pointsID[0]))
            {
                if((intersectedEdge[j][0]==pointsID[3])||(intersectedEdge[j][1]==pointsID[3]))
                {
                    int temp=pointsID[3];
                    pointsID[3]=pointsID[2];
                    pointsID[2]=temp;
                }
            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0]) || (intersectedEdge[j][1]==pointsID[0]))
                localIndex[1]=j;
            else if((intersectedEdge[j][0]==pointsID[1]) || (intersectedEdge[j][1]==pointsID[1]))
                localIndex[3]=j;
            else
                localIndex[2]=j;
        }

        Coord edgeDirec;
        edgeDirec=this->m_geometryAlgorithms->computeRestEdgeDirection(intersectedEdgeID[0])*-1;

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[6];
        if(dot>0)
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]]+1;	subTetra[0][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[2]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[2]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]];	subTetra[3][2]=intersectedPointID[localIndex[2]];	subTetra[3][3]=intersectedPointID[localIndex[3]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
        }
        else
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[2]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]];	subTetra[0][3]=intersectedPointID[localIndex[3]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]]+1;	subTetra[3][2]=intersectedPointID[localIndex[2]]+1;	subTetra[3][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
        }

        for(int i=0; i<6; i++)
        {
            if(!(m_geometryAlgorithms->checkNodeSequence(subTetra[i])))
            {
                unsigned int temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=6;
        return nbAddedTetra;
    }
    return 0;
}




} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETTOPOLOGYALGORITHMS_INL
