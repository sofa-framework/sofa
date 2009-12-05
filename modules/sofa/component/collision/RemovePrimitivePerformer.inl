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
#include <sofa/component/collision/RemovePrimitivePerformer.h>

#include <sofa/helper/Factory.inl>
#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::core::componentmodel::topology;

template <class DataTypes>
RemovePrimitivePerformer<DataTypes>::RemovePrimitivePerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i), firstClick (0)
{}


template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::start()
{
    //std::cout << " start " << std::endl;

    //    if (!selectedElem.empty())
    //       selectedElem.clear();

    if (topologicalOperation != 0)
    {
        if (firstClick)
            firstClick = false;
        else
            firstClick = true;
    }
}


template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::execute()
{
    //std::cout << " execute: " << firstClick << std::endl;

    picked=this->interactor->getBodyPicked();
    if (!picked.body) return;

    mstateCollision = dynamic_cast< core::componentmodel::behavior::MechanicalState<DataTypes>*    >(picked.body->getContext()->getMechanicalState());
    if (!mstateCollision)
    {
        std::cerr << "uncompatible MState during Mouse Interaction " << std::endl;
        return;
    }

    if (topologicalOperation == 0) // normal case, remove directly
    {
        core::CollisionElementIterator collisionElement( picked.body, picked.indexCollisionElement);

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        picked.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)   topologyChangeManager.removeItemsFromCollisionModel(collisionElement);
        picked.body=NULL;
        this->interactor->setBodyPicked(picked);
    }
    else
    {
        if (firstClick)
        {
            selectedElem.clear();
            selectedElem.resize (1);
            selectedElem[0] = picked.indexCollisionElement;

            VecIds tmp = getNeighboorElements (selectedElem);
            VecIds tmp2;
            bool end = false;

            while (!end) // Creating region of interest
            {
                tmp2 = getElementInZone (tmp);

                tmp.clear();

                if (tmp2.empty())
                    end = true;

                for (unsigned int t = 0; t<tmp2.size(); ++t)
                    selectedElem.push_back (tmp2[t]);

                tmp = getNeighboorElements (tmp2);

                tmp2.clear ();
            }
        }
        else
        {
            //std::cout << "passe la" << std::endl;

            core::CollisionElementIterator collisionElement( picked.body, picked.indexCollisionElement);

            sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
            picked.body->getContext()->get(topologyModifier);

            std::vector<int> bonFormat;
            bonFormat.resize(selectedElem.size());
            for (unsigned int i = 0; i<selectedElem.size(); ++i)
                bonFormat[i] = selectedElem[i];

            //	    if(dynamic_cast<TriangleModel*>(model)!= NULL)
            core::CollisionModel* model = collisionElement.getCollisionModel();

            // Handle Removing of topological element (from any type of topology)
            if(topologyModifier) topologyChangeManager.removeItemsFromCollisionModel(model,bonFormat );
            picked.body=NULL;
            this->interactor->setBodyPicked(picked);
        }
    }


}


/*
template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::execute()
{
}*/



template <class DataTypes>
sofa::helper::vector <unsigned int> RemovePrimitivePerformer<DataTypes>::getNeighboorElements(VecIds& elementsToTest)
{
    BaseMeshTopology* topo = picked.body->getMeshTopology();

    VecIds vertexList;
    VecIds neighboorList;

    for (unsigned int i = 0; i<elementsToTest.size(); ++i) // get list of element vertices
    {
        const BaseMeshTopology::Triangle& elem = topo->getTriangle( elementsToTest[i] );

        for (unsigned int j = 0; j<elem.size(); ++j) // Insert vertices for each element
        {
            bool Vfind = false;
            unsigned int VelemID = elem[j];

            for (unsigned int k = 0; k<vertexList.size(); ++k) // Check if not already insert
                if (vertexList[k] == VelemID)
                {
                    Vfind = true;
                    break;
                }

            if (!Vfind)
                vertexList.push_back (VelemID);
        }
    }


    for (unsigned int i = 0; i<vertexList.size(); ++i) // get list of element around previous vertices
    {
        const VecIds& elemAroundV = topo->getTrianglesAroundVertex ( vertexList[i]);

        for (unsigned int j = 0; j<elemAroundV.size(); ++j)  // Insert each element as new neighboor
        {
            bool Efind = false;
            unsigned int elemID = elemAroundV[j];

            for (unsigned int k = 0; k<neighboorList.size(); ++k) // Check if not already insert
                if (neighboorList[k] == elemID)
                {
                    Efind = true;
                    break;
                }

            if (!Efind)
                for (unsigned int k = 0; k<selectedElem.size(); ++k) // Check if not in selected list
                    if (selectedElem[k] == elemID)
                    {
                        Efind = true;
                        break;
                    }

            if (!Efind)
                neighboorList.push_back (elemID);
        }
    }

    return neighboorList;
}



template <class DataTypes>
sofa::helper::vector <unsigned int> RemovePrimitivePerformer<DataTypes>::getElementInZone(VecIds& elementsToTest)
{
    // COmpute appropriate scale from BB

    // Create zone: here a cube
    VecCoord scale_min;
    VecCoord scale_max;

    Coord center = picked.point;

    scale_min.resize( center.size());
    scale_max.resize( center.size());

    for (unsigned int i = 0; i<center.size(); ++i)
    {
        scale_min[i] = center;
        scale_max[i] = center;

        for (unsigned int j = 0; j<center.size(); ++j)
        {
            scale_max[i][j] += selectorScale/10;
            scale_min[i][j] -= selectorScale/10;
        }
    }

    typename DataTypes::VecCoord& X = *mstateCollision->getX();
    BaseMeshTopology* topo = picked.body->getMeshTopology();

    VecCoord baryCoord;
    baryCoord.resize (elementsToTest.size());

    // Compute baryCoord of lists:
    for (unsigned int i = 0; i<elementsToTest.size(); ++i)
    {
        const BaseMeshTopology::Triangle& elem = topo->getTriangle( elementsToTest[i] );

        baryCoord[i] = X[elem[0]] + X[elem[1]] + X[elem[2]];
        for (unsigned int j = 0; j<center.size(); ++j)
            baryCoord[i][j] = baryCoord[i][j]/3;

    }


    VecIds elemInside;
    // Test if points are in zone
    for (unsigned int i = 0; i<elementsToTest.size(); ++i)
    {
        bool inSide = true;

        for (unsigned int j = 0; j<center.size(); ++j)
        {
            for (unsigned int k = 0; k<center.size(); ++k)
            {
                if (baryCoord[i][k] <= scale_min[j][k]) // check min borns
                {
                    inSide = false;
                    break;
                }
                if (baryCoord[i][k] >= scale_max[j][k])
                {
                    inSide = false;
                    break;
                }
            }
        }

        if (inSide)
            elemInside.push_back (elementsToTest[i]);
    }

    return elemInside;
}




template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::end()
{
    //std::cout << "RemovePrimitivePerformer::end()" << std::endl;
    //	firstClick = true;
}



template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::draw()
{
    if (picked.body == NULL) { std::cout << "sort 1" << std::endl; return;}

    if (mstateCollision == NULL) {std::cout << "sort 2" << std::endl; return;}


    typename DataTypes::VecCoord& X = *mstateCollision->getX();
    core::componentmodel::topology::BaseMeshTopology* topo = picked.body->getMeshTopology();

    glDisable(GL_LIGHTING);
    glColor3f(0.2,0.8,0.8);
    glBegin(GL_TRIANGLES);

    for (unsigned int i=0; i<selectedElem.size(); ++i)
    {
        const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = topo->getTriangle(selectedElem[i]);

        for (unsigned int j = 0; j<3; j++)
        {
            Coord coordP = X[elem[j]];
            glVertex3d(coordP[0], coordP[1], coordP[2]);
        }
    }
    glEnd();
}


}
}
}

