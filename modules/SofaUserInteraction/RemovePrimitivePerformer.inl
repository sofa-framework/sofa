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
#include <SofaUserInteraction/RemovePrimitivePerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
RemovePrimitivePerformer<DataTypes>::RemovePrimitivePerformer(BaseMouseInteractor *i)
    :TInteractionPerformer<DataTypes>(i)
    ,firstClick (0)
    ,surfaceOnVolume(false)
    ,volumeOnSurface(false)
    ,topo_curr(NULL)
{}


/// Functions called in framework of the mouse Interactor
//***************************************************************************************************************

template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::start()
{
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
    // - STEP 1: Get body picked and Mstate associated
    picked=this->interactor->getBodyPicked();
    if (!picked.body) return;

    mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*    >(picked.body->getContext()->getMechanicalState());
    if (!mstateCollision)
    {
        msg_warning("RemovePrimitivePerformer") << "Incompatible Mechanical State during Mouse Interaction ";
        return;
    }

    // - STEP 1: Checking type of operation
    if (topologicalOperation == 0) // normal case, remove directly one element
    {
        core::CollisionElementIterator collisionElement( picked.body, picked.indexCollisionElement);
        core::CollisionModel* model = collisionElement.getCollisionModel();

        sofa::core::topology::TopologyModifier* topologyModifier;
        picked.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
            topologyChangeManager.removeItemsFromCollisionModel(model, (int)picked.indexCollisionElement);

        picked.body=NULL;
        this->interactor->setBodyPicked(picked);
    }
    else // second case remove a zone of element
    {
        if (firstClick) // first click detected => creation of the zone
        {
            if (!createElementList())
                return;
        }
        else // second clic removing zone stored in selectedElem
        {
            if (selectedElem.empty())
                return;

            core::CollisionElementIterator collisionElement( picked.body, picked.indexCollisionElement);

            sofa::core::topology::TopologyModifier* topologyModifier;
            picked.body->getContext()->get(topologyModifier);

            // Problem of type takeng by functions called: Converting selectedElem <unsigned int> in <int>
            helper::vector<int> ElemList_int;
            ElemList_int.resize(selectedElem.size());
            for (unsigned int i = 0; i<selectedElem.size(); ++i)
                ElemList_int[i] = selectedElem[i];

            // Creating model of collision
            core::CollisionModel::SPtr model;
            if (surfaceOnVolume) // In the case of deleting a volume from a surface an volumique collision model is needed (only tetra available for the moment)
            {
#if 0
                model = sofa::core::objectmodel::New<TetrahedronModel>();
                //model->setContext(topo_curr->getContext());
                topo_curr->getContext()->addObject(model);
#endif
            }
            else // other cases, collision model from pick is taken
            {
                model = collisionElement.getCollisionModel();
            }

            // Handle Removing of topological element (from any type of topology)
            if(topologyModifier) topologyChangeManager.removeItemsFromCollisionModel(model.get(),ElemList_int );
            picked.body=NULL;
            this->interactor->setBodyPicked(picked);

            if (surfaceOnVolume) // In the case of deleting a volume from a surface an volumique collision model is needed (only tetra available for the moment)
            {
#if 0
                topo_curr->getContext()->removeObject(model);
#endif
            }
            selectedElem.clear();
        }
    }
}


template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::end()
{
    dmsg_info("RemovePrimitivePerfomer") << " end()" ;
}



//***************************************************************************************************************
// Internal functions

// ** Creating a list of elements concerned by the removal operation **
template <class DataTypes>
bool RemovePrimitivePerformer<DataTypes>::createElementList()
{
    // - STEP 1: Looking for current topology type
    topo_curr = picked.body->getContext()->getMeshTopology();
    if (topo_curr->getNbHexahedra())
        topoType = sofa::core::topology::HEXAHEDRON;
    else if (topo_curr->getNbTetrahedra())
        topoType = sofa::core::topology::TETRAHEDRON;
    else if (topo_curr->getNbQuads())
        topoType = sofa::core::topology::QUAD;
    else if (topo_curr->getNbTriangles())
        topoType = sofa::core::topology::TRIANGLE;
    else
    {
        msg_error("RemovePrimitivePerformer") << "No topology has been found." ;
        return false;
    }

    // Initialization of first element
    selectedElem.clear();
    selectedElem.resize (1);
    selectedElem[0] = picked.indexCollisionElement;

    // - STEP 2: Looking for type of zone to remove
    if (!volumicMesh) // Surfacique case
    {
        volumeOnSurface = false;
        sofa::core::topology::TopologyObjectType topoTypeTmp = topoType;

        // - STEP 3: Looking for tricky case
        if (topoType == sofa::core::topology::TETRAHEDRON || topoType == sofa::core::topology::HEXAHEDRON) // special case: removing a surface volume on the mesh (tetra only for the moment)
        {
            // looking for mapping VolumeToSurface
            simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());
            std::vector< core::objectmodel::BaseObject * > listObject;
            node_curr->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::SearchRoot);

            for(unsigned int i=0; i<listObject.size(); ++i) // loop on all components to find mapping
            {
                sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
                if (topoMap)
                {
                    // Mapping found: 1- looking for volume, 2- looking for surface element on border, 3- looking for correspondant ID element in surfacique mesh
                    const sofa::core::topology::BaseMeshTopology::TrianglesInTetrahedron& tetraTri = topo_curr->getTrianglesInTetrahedron(selectedElem[0]);

                    int volTmp = -1;
                    std::map<unsigned int, unsigned int> MappingMap = topoMap->getGlob2LocMap();
                    std::map<unsigned int, unsigned int>::iterator it;

                    for (unsigned int j = 0; j<4; ++j)
                    {
                        it = MappingMap.find (tetraTri[j]);
                        if ( it != MappingMap.end())
                        {
                            volTmp = (*it).second;
                            break;
                        }
                    }

                    if (volTmp == -1)
                    {
                        msg_error("RemovePrimitivePerformer") << "Problem while looking for corresponding element on surface mesh." ;
                        return false;
                    }

                    // Surface element has been found, computation will be done on surfacique mesh => switch temporary all variables to surface
                    selectedElem[0] = (unsigned int)volTmp;
                    volumeOnSurface = true;
                    topo_curr = topoMap->getTo();
                    topoType = sofa::core::topology::TRIANGLE;
                }
            }

            if (!volumeOnSurface)
            {
                msg_warning("RemovePrimitivePerformer") << "Trying to remove a volume at the surface of the mesh without using "
                                 "mapping volume to surface mesh. This case is not supported." ;
                return false;
            }
        }


        // - STEP 4: Loop on getNeighboorElements and getElementInZone until no more nighboor are in zones
        // Initialization
        bool end = false;
        VecIds tmp = getNeighboorElements (selectedElem);
        VecIds tmp2;

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


        // - STEP 5: Postprocessing: zone using surface element has been found, extract volumes behind (small error on boundary regarding barycentric points)
        if (volumeOnSurface)
        {
            // Get dofs on surface
            for (unsigned int i = 0; i<selectedElem.size(); ++i)
            {
                helper::vector<unsigned int> elem;

                switch ( topoType ) // Get surfacique elements as array of vertices
                {
                case sofa::core::topology::QUAD:
                {
                    const sofa::core::topology::BaseMeshTopology::Quad& quad = topo_curr->getQuad(selectedElem[i]);
                    elem.resize(4);
                    for (unsigned int j = 0; j<4; ++j)
                        elem[j] = quad[j];
                    break;
                }
                case sofa::core::topology::TRIANGLE:
                {
                    const sofa::core::topology::BaseMeshTopology::Triangle& tri = topo_curr->getTriangle(selectedElem[i]);
                    elem.resize(3);
                    for (unsigned int j = 0; j<3; ++j)
                        elem[j] = tri[j];
                    break;
                }
                default:
                    break;
                }

                // Pattern fill vector without redundancy
                for (unsigned int j = 0; j<elem.size(); ++j)
                {
                    bool dofFind = false;
                    unsigned int Selem = elem[j];

                    for (unsigned int k = 0; k<tmp2.size(); ++k)
                        if (tmp2[j] == Selem)
                        {
                            dofFind = true;
                            break;
                        }

                    if (!dofFind)
                        tmp2.push_back(Selem);
                }
            }

            // Switching variables to initial topology (topotype, topology) clear list of surfacique elements selected
            topo_curr = picked.body->getMeshTopology();
            topoType = topoTypeTmp;
            selectedElem.clear();

            // Get Volumique elements from list of vertices in tmp2
            for (unsigned int i = 0; i<tmp2.size(); ++i)
            {
                helper::vector<unsigned int> elem;

                switch ( topoType )
                {
                case sofa::core::topology::HEXAHEDRON:
                {
                    const sofa::core::topology::BaseMeshTopology::HexahedraAroundVertex& hexaV = topo_curr->getHexahedraAroundVertex(tmp2[i]);
                    for (unsigned int j = 0; j<hexaV.size(); ++j)
                        elem.push_back(hexaV[j]);

                    break;
                }
                case sofa::core::topology::TETRAHEDRON:
                {
                    const sofa::core::topology::BaseMeshTopology::TetrahedraAroundVertex& tetraV = topo_curr->getTetrahedraAroundVertex(tmp2[i]);
                    for (unsigned int j = 0; j<tetraV.size(); ++j)
                        elem.push_back(tetraV[j]);

                    break;
                }
                default:
                    break;
                }

                // Pattern fill vector without redundancy
                for (unsigned int j = 0; j<elem.size(); ++j)
                {
                    bool Vfind = false;
                    unsigned int VelemID = elem[j];

                    for (unsigned int k = 0; k<selectedElem.size(); ++k) // Check if not already insert
                        if (selectedElem[k] == VelemID)
                        {
                            Vfind = true;
                            break;
                        }

                    if (!Vfind)
                        selectedElem.push_back (VelemID);
                }
            }
        }

    }
    else // - STEP 2: Volumique case
    {
        surfaceOnVolume = false;

        // - STEP 3: Looking for tricky case
        if (topoType == sofa::core::topology::TRIANGLE || topoType == sofa::core::topology::QUAD) // Special case: removing a volumique zone on the mesh while starting at the surface
        {
            // looking for mapping VolumeToSurface
            simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());
            std::vector< core::objectmodel::BaseObject * > listObject;
            node_curr->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::Local);

            for(unsigned int i=0; i<listObject.size(); ++i) // loop on all components to find mapping (only tetra for the moment)
            {
                sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
                if (topoMap)
                {
                    // Mapping found: 1- get surface element ID in volumique topology, 2- get volume element ID behind surface element, 3- switching all variables to volumique case
                    unsigned int volTmp = (topoMap->getLoc2GlobVec()).getValue()[selectedElem[0]];
                    topo_curr = topoMap->getFrom();
                    selectedElem[0] = topo_curr->getTetrahedraAroundTriangle(volTmp)[0];
                    surfaceOnVolume = true;
                    topoType = sofa::core::topology::TETRAHEDRON;
                }
            }

            if (!surfaceOnVolume)
            {
                msg_warning("RemovePrimitivePerformer") << "Trying to remove a volume using a surfacique mesh without mapping to volume mesh." ;
                return false;
            }
        }

        // - STEP 4: Loop on getNeighboorElements and getElementInZone until no more nighboor are in zones
        // Initialization
        bool end = false;
        VecIds tmp = getNeighboorElements (selectedElem);
        VecIds tmp2;

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

    return true;
}



// ** Return a vector of elements directly neighboor of a given list of elements **
template <class DataTypes>
sofa::helper::vector <unsigned int> RemovePrimitivePerformer<DataTypes>::getNeighboorElements(VecIds& elementsToTest)
{
    VecIds vertexList;
    VecIds neighboorList;


    // - STEP 1: get list of element vertices
    for (unsigned int i = 0; i<elementsToTest.size(); ++i)
    {
        helper::vector<unsigned int> elem;

        switch ( topoType ) // Get element as array of vertices
        {
        case sofa::core::topology::HEXAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Hexa& hexa = topo_curr->getHexahedron(elementsToTest[i]);
            elem.resize(8);
            for (unsigned int j = 0; j<8; ++j)
                elem[j] = hexa[j];
            break;
        }
        case sofa::core::topology::TETRAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Tetra& tetra = topo_curr->getTetrahedron(elementsToTest[i]);
            elem.resize(4);
            for (unsigned int j = 0; j<4; ++j)
                elem[j] = tetra[j];
            break;
        }
        case sofa::core::topology::QUAD:
        {
            const sofa::core::topology::BaseMeshTopology::Quad& quad = topo_curr->getQuad(elementsToTest[i]);
            elem.resize(4);
            for (unsigned int j = 0; j<4; ++j)
                elem[j] = quad[j];
            break;
        }
        case sofa::core::topology::TRIANGLE:
        {
            const sofa::core::topology::BaseMeshTopology::Triangle& tri = topo_curr->getTriangle(elementsToTest[i]);
            elem.resize(3);
            for (unsigned int j = 0; j<3; ++j)
                elem[j] = tri[j];
            break;
        }
        default:
            break;
        }

        // Pattern fill vector without redundancy
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

    // - STEP 2: get list of element around vertices previously obtained
    for (unsigned int i = 0; i<vertexList.size(); ++i)
    {
        VecIds elemAroundV;

        switch ( topoType ) // Get elements around vertices as array of ID
        {
        case sofa::core::topology::HEXAHEDRON:
        {
            elemAroundV = topo_curr->getHexahedraAroundVertex (vertexList[i]);
            break;
        }
        case sofa::core::topology::TETRAHEDRON:
        {
            elemAroundV = topo_curr->getTetrahedraAroundVertex (vertexList[i]);
            break;
        }
        case sofa::core::topology::QUAD:
        {
            elemAroundV = topo_curr->getQuadsAroundVertex (vertexList[i]);
            break;
        }
        case sofa::core::topology::TRIANGLE:
        {
            elemAroundV = topo_curr->getTrianglesAroundVertex (vertexList[i]);
            break;
        }
        default:
            break;
        }

        // Pattern fill vector without redundancy + checking not insert in input selectedElem
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


// ** Function testing if elements are in the range of a given zone **
template <class DataTypes>
sofa::helper::vector <unsigned int> RemovePrimitivePerformer<DataTypes>::getElementInZone(VecIds& elementsToTest)
{
    // - STEP 0: Compute appropriate scale from BB:  selectorScale = 100 => zone = all mesh
    defaulttype::Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    core::objectmodel::BaseNode* root = mstateCollision->getContext()->toBaseNode();
    if (root) root = root->getRoot();
    if (root) { sceneMinBBox = root->f_bbox.getValue().minBBox(); sceneMaxBBox = root->f_bbox.getValue().maxBBox(); }
    else      { sceneMinBBox = mstateCollision->getContext()->f_bbox.getValue().minBBox(); sceneMaxBBox = mstateCollision->getContext()->f_bbox.getValue().maxBBox(); }
    Real BB_size = (Real)(sceneMaxBBox - sceneMinBBox).norm();
    if (BB_size == 0)
    {
        msg_info("RemovePrimitivePerformer") << "While computing Boundingbox size, size return null." ;
        BB_size = 1; // not to crash program
    }
    Real zone_size = (Real)(BB_size*selectorScale)/200;
    Real dist;
    Coord center = picked.point;

    // - STEP 2: Compute baryCoord of elements in list:
    const VecCoord& X = mstateCollision->read(core::ConstVecCoordId::position())->getValue();

    VecCoord baryCoord;
    baryCoord.resize (elementsToTest.size());

    for (unsigned int i = 0; i<elementsToTest.size(); ++i)
    {
        unsigned int N = 1;

        switch ( topoType ) // get element as array of vertices and sum the coordinates
        {
        case sofa::core::topology::HEXAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Hexa& hexa = topo_curr->getHexahedron(elementsToTest[i]);
            baryCoord[i] = X[hexa[0]] + X[hexa[1]] + X[hexa[2]] + X[hexa[3]] +
                    X[hexa[4]] + X[hexa[5]] + X[hexa[6]] + X[hexa[7]];
            N = 8;

            break;
        }
        case sofa::core::topology::TETRAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Tetra& tetra = topo_curr->getTetrahedron(elementsToTest[i]);
            baryCoord[i] = X[tetra[0]] + X[tetra[1]] + X[tetra[2]] + X[tetra[3]];
            N = 4;

            break;
        }
        case sofa::core::topology::QUAD:
        {
            const sofa::core::topology::BaseMeshTopology::Quad& quad = topo_curr->getQuad(elementsToTest[i]);
            baryCoord[i] = X[quad[0]] + X[quad[1]] + X[quad[2]] + X[quad[3]];
            N = 4;

            break;
        }
        case sofa::core::topology::TRIANGLE:
        {
            const sofa::core::topology::BaseMeshTopology::Triangle& tri = topo_curr->getTriangle(elementsToTest[i]);
            baryCoord[i] = X[tri[0]] + X[tri[1]] + X[tri[2]];
            N = 3;

            break;
        }
        default:
            break;
        }

        for (unsigned int j = 0; j<center.size(); ++j) // divided each coordinate by N (number of vertices)
            baryCoord[i][j] = baryCoord[i][j]/N;

    }


    VecIds elemInside;
    // - STEP 3: Test if barycentric points are inside the zone
    for (unsigned int i = 0; i<elementsToTest.size(); ++i)
    {
        //compute distance from barycenter to center zone
        dist = (baryCoord[i] - center).norm();

        if (dist < zone_size)
            elemInside.push_back (elementsToTest[i]);
    }

    return elemInside;
}



//***************************************************************************************************************

template <class DataTypes>
void RemovePrimitivePerformer<DataTypes>::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    if (picked.body == NULL) return;

    if (mstateCollision == NULL) return;


    const VecCoord& X = mstateCollision->read(core::ConstVecCoordId::position())->getValue();
    //core::topology::BaseMeshTopology* topo = picked.body->getMeshTopology();

    glDisable(GL_LIGHTING);
    glColor3f(0.3f,0.8f,0.3f);


    if (topoType == sofa::core::topology::QUAD || topoType == sofa::core::topology::HEXAHEDRON)
        glBegin (GL_QUADS);
    else
        glBegin (GL_TRIANGLES);


    for (unsigned int i=0; i<selectedElem.size(); ++i)
    {
        helper::vector<unsigned int> elem;

        switch ( topoType )
        {
        case sofa::core::topology::HEXAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Hexa& hexa = topo_curr->getHexahedron(selectedElem[i]);
            Coord coordP[8];

            for (unsigned int j = 0; j<8; j++)
                coordP[j] = X[hexa[j]];

            for (unsigned int j = 0; j<8; ++j)
            {
                glVertex3d(coordP[j][0], coordP[j][1], coordP[j][2]);
                glVertex3d(coordP[(j+1)%4][0], coordP[(j+1)%4][1], coordP[(j+1)%4][2]);
                glVertex3d(coordP[(j+2)%4][0], coordP[(j+2)%4][1], coordP[(j+2)%4][2]);
                glVertex3d(coordP[(j+3)%4][0], coordP[(j+3)%4][1], coordP[(j+3)%4][2]);
            }
            break;
        }
        case sofa::core::topology::TETRAHEDRON:
        {
            const sofa::core::topology::BaseMeshTopology::Tetra& tetra = topo_curr->getTetrahedron(selectedElem[i]);
            Coord coordP[4];

            for (unsigned int j = 0; j<4; j++)
                coordP[j] = X[tetra[j]];

            for (unsigned int j = 0; j<4; ++j)
            {
                glVertex3d(coordP[j][0], coordP[j][1], coordP[j][2]);
                glVertex3d(coordP[(j+1)%4][0], coordP[(j+1)%4][1], coordP[(j+1)%4][2]);
                glVertex3d(coordP[(j+2)%4][0], coordP[(j+2)%4][1], coordP[(j+2)%4][2]);
            }
            break;
        }
        case sofa::core::topology::QUAD:
        {
            const sofa::core::topology::BaseMeshTopology::Quad& quad = topo_curr->getQuad(selectedElem[i]);

            for (unsigned int j = 0; j<4; j++)
            {
                Coord coordP = X[quad[j]];
                glVertex3d(coordP[0], coordP[1], coordP[2]);
            }
            break;
        }
        case sofa::core::topology::TRIANGLE:
        {
            const sofa::core::topology::BaseMeshTopology::Triangle& tri = topo_curr->getTriangle(selectedElem[i]);

            for (unsigned int j = 0; j<3; j++)
            {
                Coord coordP = X[tri[j]];
                glVertex3d(coordP[0] * 1.001, coordP[1] * 1.001, coordP[2] * 1.001);
            }
            for (unsigned int j = 0; j<3; j++)
            {
                Coord coordP = X[tri[j]];
                glVertex3d(coordP[0] * 0.999, coordP[1] * 0.999, coordP[2] * 0.999);
            }

            break;
        }
        default:
            break;
        }



    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}


}
}
}

