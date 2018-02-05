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

#include "SculptBodyPerformer.h"
#include <SofaBaseCollision/CubeModel.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaBaseMechanics/SubsetMapping.h>
#include <SofaEngine/MergePoints.h>

namespace plugins
{

namespace pim
{

using namespace sofa::component::mapping;
using namespace sofa::component::engine;
using namespace sofa::component::forcefield;
using namespace sofa::component::constraint;
using namespace sofa::component::container;
using namespace sofa::component::visualmodel;
using namespace sofa::core::behavior;
using namespace sofa::core;


/// Computes the neigborhood of the picked point
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::computeNeighborhood()
{
    sofa::core::topology::BaseMeshTopology::VerticesAroundVertex vav;
    vertexNeighborhood.clear();

    for( std::set<unsigned int>::const_iterator iter = vertexInInfluenceZone.begin(); iter != vertexInInfluenceZone.end(); ++iter )
    {
        vav = picked.body->getMeshTopology()->getVerticesAroundVertex(*iter);
        for (unsigned int i=0; i<vav.size(); ++i)
        {
            if (alreadyCheckedVertex.find(vav[i]) == alreadyCheckedVertex.end())
            {
                vertexNeighborhood.insert(vav[i]);
            }
        }
    }
}

/// Gets the nearest point from the picked collision element (Triangle)
template <class DataTypes>
unsigned int SculptBodyPerformer<DataTypes>::computeNearestVertexFromPickedPoint()
{
    const BaseMeshTopology::Triangle& pickedTriangle = picked.body->getMeshTopology()->getTriangle(picked.indexCollisionElement);
    const typename DataTypes::VecCoord& X = *mstateCollision->getX();

    unsigned int nearestVertexFromPickedPoint = pickedTriangle[0];
    if ((X[nearestVertexFromPickedPoint] - picked.point).norm() > (X[pickedTriangle[1]] - picked.point).norm())
    {
        nearestVertexFromPickedPoint = pickedTriangle[1];
    }
    if ((X[nearestVertexFromPickedPoint] - picked.point).norm() > (X[pickedTriangle[2]] - picked.point).norm())
    {
        nearestVertexFromPickedPoint = pickedTriangle[2];
    }
    return nearestVertexFromPickedPoint;
}

/// Computes the normal of the point "index"
template <class DataTypes>
typename DataTypes::Coord SculptBodyPerformer<DataTypes>::computeNormal(const unsigned int& index)
{
    const typename DataTypes::VecCoord& X = *mstateCollision->getX();
    const BaseMeshTopology::TrianglesAroundVertex& tav = picked.body->getMeshTopology()->getTrianglesAroundVertex(index);
    Coord normal = Coord(0,0,0);

    for (unsigned int j=0; j<tav.size(); ++j)
    {
        const BaseMeshTopology::Triangle& t = picked.body->getMeshTopology()->getTriangle(tav[j]);
        const Coord p0 = X[t[0]];
        const Coord p1 = X[t[1]];
        const Coord p2 = X[t[2]];
        normal += cross(p1-p0, p2-p0);
    }
    normal.normalize();
    return normal;
}

/// Update the position of each vertex according to the corresponding action (inflate/deflate)
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::updatePosition(const unsigned int& index, const typename DataTypes::Coord& normal, const double& W)
{
    typename DataTypes::VecCoord& X = *mstateCollision->getX();
    if (checkedInflate)
    {
        X[index] += normal * W * (force);
    }
    else
    {
        if (checkedDeflate)
        {
            X[index] -= normal * W * (force);
        }
    }
}

/// Inflate/deflate from the picked point
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::inflate()
{
    typename DataTypes::VecCoord& X = *mstateCollision->getX();

    typename DataTypes::Real dist, W;

    // Gets the diagonal of the bounding box
    CubeModel* cModel = dynamic_cast<CubeModel*>(picked.body->getFirst());
    Cube bbox = Cube(cModel,0);
    typename DataTypes::Real bboxDiagonal = (typename DataTypes::Real)((bbox.minVect() - bbox.maxVect()).norm()/4);
    // Compute the ratio to inflate/deflate
    typename DataTypes::Real ratio = bboxDiagonal * (typename DataTypes::Real) (scale/99);

    // Gets the nearest point from the picked collision element (Triangle)
    unsigned int nearestVertexFromPickedPoint = computeNearestVertexFromPickedPoint();

    vertexNeighborhood.insert(nearestVertexFromPickedPoint);
    alreadyCheckedVertex.clear();
    do
    {
        vertexInInfluenceZone.clear();
        for( std::set<unsigned int>::const_iterator iter = vertexNeighborhood.begin(); iter != vertexNeighborhood.end(); ++iter )
        {
            dist = (X[*iter]-picked.point).norm();
            if (dist < ratio)
            {
                vertexInInfluenceZone.insert(*iter);
                W = pow(pow(1-dist/ratio,2),2)*(4*dist/ratio+1);
                // Computes the normal of the point "index"
                Coord normal = computeNormal(*iter);
                // Update the position of each vertex according to the corresponding action (inflate/deflate)
                updatePosition(*iter, normal, W);
            }
            alreadyCheckedVertex.insert(*iter);
        }
        computeNeighborhood();
    }
    while(!vertexInInfluenceZone.empty());
}

/// stops object movements removing the corresponding solver
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::stopSimulation()
{
    root->getTreeObjects<sofa::core::behavior::OdeSolver>(&solvers);
    root->getTreeObjects<sofa::component::linearsolver::CGLinearSolver<sofa::component::linearsolver::GraphScatteredMatrix,sofa::component::linearsolver::GraphScatteredVector> >(&linear);

    for (unsigned int i=0; i<solvers.size(); ++i)
    {
        bc = solvers[i]->getContext();
        bc->removeObject(solvers[i]);
        bc->removeObject(linear[i]);
    }
}

///save current state and mesh for intersect layers
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::saveCurrentStateAndMesh()
{
    BaseMeshTopology* mesh = dynamic_cast<BaseMeshTopology*>(mstateCollision->getContext()->getTopology());
    cmi = new ComputeMeshIntersection<DataTypes>();
    cmi->d_muscleLayerVertex.setValue(*mstateCollision->getX());
    cmi->d_muscleLayerTriangles.setValue(mesh->getTriangles());
}

/// action performed at the begining of the mouse interaction
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::start()
{
    picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.body->getContext()->getMechanicalState());
        if (!mstateCollision)
        {
            std::cerr << "incompatible MState during Mouse Interaction " << std::endl;
            return;
        }

        if (startSculpt)
        {
            //stop the simualtion to be able to sculpt
            stopSimulation();

            //save current state and mesh for intersection layers
            saveCurrentStateAndMesh();

            startSculpt=false;
        }
        // save state for undo operations
//              eventManager.saveMechanicalState(mstateCollision);
    }
}

/// action performed during the mouse interaction
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::execute()
{
    picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.body->getContext()->getMechanicalState());
        if (!mstateCollision)
        {
            std::cerr << "incompatible MState during Mouse Interaction " << std::endl;
            return;
        }
        inflate();
    }
}


/// Tetra mesh generator
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::createTetraVolume(BaseMeshTopology* mesh)
{
    cgal::MeshGenerationFromPolyhedron<DataTypes>* CGALmeshStuffed = new cgal::MeshGenerationFromPolyhedron<DataTypes>();
    CGALmeshStuffed->f_X0.setValue(cmi->d_intersectionVertex.getValue());
    CGALmeshStuffed->f_triangles.setValue(cmi->d_intersectionTriangles.getValue());
    CGALmeshStuffed->facetAngle.setValue(30);
    CGALmeshStuffed->facetSize.setValue(0.5);
    CGALmeshStuffed->facetApproximation.setValue(0.05);
    CGALmeshStuffed->cellRatio.setValue(2);
    CGALmeshStuffed->cellSize.setValue(0.5);
    CGALmeshStuffed->odt.setValue(true);
    CGALmeshStuffed->odt_max_it.setValue(200);
    CGALmeshStuffed->perturb.setValue(true);
    CGALmeshStuffed->exude_max_time.setValue(20);
    CGALmeshStuffed->init();
    CGALmeshStuffed->update();

    for (unsigned int i=0; i<CGALmeshStuffed->f_newX0.getValue().size(); ++i)
    {
        mesh->addPoint(CGALmeshStuffed->f_newX0.getValue()[i][0], CGALmeshStuffed->f_newX0.getValue()[i][1],
                CGALmeshStuffed->f_newX0.getValue()[i][2]);
    }

    for (unsigned int i=0; i<CGALmeshStuffed->f_tetrahedra.getValue().size(); ++i)
    {
        mesh->addTetra(CGALmeshStuffed->f_tetrahedra.getValue()[i][0], CGALmeshStuffed->f_tetrahedra.getValue()[i][1],
                CGALmeshStuffed->f_tetrahedra.getValue()[i][2], CGALmeshStuffed->f_tetrahedra.getValue()[i][3]);
    }
}

///compute intersection between the original and the modified layers
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::ComputeIntersectonLayer()
{
    BaseMeshTopology* mstateCollisionMesh = dynamic_cast<BaseMeshTopology*>(mstateCollision->getContext()->getTopology());
    cmi->d_fatLayerVertex.setValue(*mstateCollision->getX());
    cmi->d_fatLayerTriangles.setValue(mstateCollisionMesh->getTriangles());
    cmi->init();
    cmi->update();
}

///create the new collision model resulting from the union between the object and the fat
template <class DataTypes>
sofa::simulation::tree::GNode* SculptBodyPerformer<DataTypes>::createCollisionNode(sofa::simulation::tree::GNode* parentNode)
{
    sofa::simulation::tree::GNode* collisionNode = dynamic_cast<sofa::simulation::tree::GNode*>(mstateCollision->getContext());

    parentNode->addChild(collisionNode);

    MeshLoader* meshloader;
    collisionNode->get(meshloader, sofa::core::objectmodel::BaseContext::SearchDown);
    if (meshloader != NULL)
    {
        collisionNode->removeObject(meshloader);
    }

    BaseMeshTopology* mesh = collisionNode->getMeshTopology();
    BaseMeshTopology::SeqTriangles triangles = mesh->getTriangles();

    mesh->clear();

    for (unsigned int i=0; i<(*mstateCollision->getX()).size(); ++i)
    {
        mesh->addPoint((*mstateCollision->getX())[i][0], (*mstateCollision->getX())[i][1],(*mstateCollision->getX())[i][2]);
    }

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        mesh->addTriangle(triangles[i][0], triangles[i][1],triangles[i][2]);
    }

    BaseMapping* mapping;
    collisionNode->get(mapping, sofa::core::objectmodel::BaseContext::SearchDown);
    collisionNode->removeObject(mapping);

    BaseMapping* newMapping = new BarycentricMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >(dynamic_cast<MechanicalState<Vec3Types>* >(parentNode->getMechanicalState()), dynamic_cast<MechanicalState<Vec3Types>*>(collisionNode->getMechanicalState()));
    collisionNode->addObject(newMapping);

    return collisionNode;
}

///create the new collision model resulting from the union between the object and the fat
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::createVisualNode(sofa::simulation::tree::GNode* parentNode)
{
    OglModel* oglModel;
    visualRoot->get(oglModel, sofa::core::objectmodel::BaseContext::SearchDown);

    for (unsigned int i=0; i<(*mstateCollision->getX()).size(); ++i)
    {
        (*oglModel->getVecX())[i] = (*mstateCollision->getX())[i];
    }

    Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> >* mapping;
    visualRoot->get(mapping, sofa::core::objectmodel::BaseContext::SearchDown);

    oglModel->getContext()->removeObject(mapping);

    BaseMapping* newMapping = new IdentityMapping<Mapping<State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >(dynamic_cast<State<Vec3dTypes>*>(parentNode->getMechanicalState()), (MappedModel<ExtVec3fTypes>*)oglModel);

    oglModel->getContext()->addObject(newMapping);
}


///allows to manage the object and the fat separately
template <class DataTypes>
sofa::simulation::tree::GNode* SculptBodyPerformer<DataTypes>::createSubsetMapping(sofa::simulation::tree::GNode* parentNode, MechanicalState<DataTypes>* subsetState,
        BaseMeshTopology* subsetTopology, double m, double ymodulus, double damping)
{
    sofa::simulation::tree::GNode* node = new sofa::simulation::tree::GNode;
    parentNode->addChild(node);

    BaseMeshTopology* mesh = new TetrahedronSetTopologyContainer();

    for (unsigned int i=0; i<(*subsetState->getX()).size(); ++i)
    {
        mesh->addPoint((*subsetState->getX())[i][0], (*subsetState->getX())[i][1], (*subsetState->getX())[i][2]);
    }
    for (int i=0; i<subsetTopology->getNbTetrahedra(); ++i)
    {
        mesh->addTetra(subsetTopology->getTetrahedron(i)[0], subsetTopology->getTetrahedron(i)[1], subsetTopology->getTetrahedron(i)[2], subsetTopology->getTetrahedron(i)[3]);
    }

    node->addObject(mesh);

    sofa::component::container::MechanicalObject<DataTypes>* mstate = new sofa::component::container::MechanicalObject<DataTypes>();
    mstate->setName("fat");
    node->addObject(mstate);

    TetrahedronSetGeometryAlgorithms<DataTypes>* tsga = new TetrahedronSetGeometryAlgorithms<DataTypes>();
    node->addObject(tsga);

    sofa::component::mass::DiagonalMass<DataTypes, double>* mass = new sofa::component::mass::DiagonalMass<DataTypes, double>();
    mass->setMassDensity(m);
    node->addObject(mass);

    sofa::component::forcefield::TetrahedralCorotationalFEMForceField<DataTypes>* forcefield = new sofa::component::forcefield::TetrahedralCorotationalFEMForceField<DataTypes>();
    forcefield->setYoungModulus(ymodulus);
    forcefield->setPoissonRatio(damping);
    forcefield->setComputeGlobalMatrix(false);
    forcefield->setMethod(1);
    node->addObject(forcefield);

    BaseMapping* mapping = new SubsetMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >(dynamic_cast<MechanicalState<Vec3Types>*>(parentNode->getMechanicalState()), (MechanicalState<Vec3Types>*)mstate);
    node->addObject(mapping);

    return node;
}

///create the contact surface to be able to fix the fat to the object
template <class DataTypes>
MechanicalState<DataTypes>* SculptBodyPerformer<DataTypes>::createContactSurfaceMapping(sofa::simulation::tree::GNode* parentNode)
{
    sofa::simulation::tree::GNode* node = new sofa::simulation::tree::GNode;
    parentNode->addChild(node);

    BaseMeshTopology* mesh = new MeshTopology();

    for (unsigned int i=0; i<cmi->d_intersectionVertex.getValue().size(); i+=2)
    {
        mesh->addPoint(cmi->d_intersectionVertex.getValue()[i][0], cmi->d_intersectionVertex.getValue()[i][1], cmi->d_intersectionVertex.getValue()[i][2]);
    }

    for (unsigned int i=0; i<cmi->d_intersectionTriangles.getValue().size(); i+=2)
    {
        mesh->addTriangle(cmi->d_intersectionTriangles.getValue()[i][0]/2, cmi->d_intersectionTriangles.getValue()[i][1]/2, cmi->d_intersectionTriangles.getValue()[i][2]/2);
    }

    node->addObject(mesh);

    sofa::component::container::MechanicalObject<DataTypes>* mstate = new sofa::component::container::MechanicalObject<DataTypes>();
    node->addObject(mstate);

    BaseMapping* mapping = new BarycentricMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >(dynamic_cast<MechanicalState<Vec3Types>*>(parentNode->getMechanicalState()), (MechanicalState<Vec3Types>*)mstate);
    node->addObject(mapping);

    return mstate;
}

///create the new state compose by the object plus the fat
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::CreateNewMechanicalState(MechanicalState<DataTypes>* fatmstate, BaseMeshTopology* fatMesh)
{
    //get state and topology of the sculpted object
    sofa::simulation::tree::GNode* collisionNode = dynamic_cast<sofa::simulation::tree::GNode*>(mstateCollision->getContext());
    sofa::simulation::tree::GNode* behaviorNode = dynamic_cast<sofa::simulation::tree::GNode*>(collisionNode->getParent());
    MechanicalState<DataTypes>* objectMsState = dynamic_cast<MechanicalState<DataTypes>*>(behaviorNode->getMechanicalState());
    BaseMeshTopology* objecTopology = dynamic_cast<BaseMeshTopology*>(behaviorNode->getMeshTopology());

    root->removeChild(behaviorNode);
    behaviorNode->removeChild(collisionNode);

    sofa::simulation::tree::GNode* node = new sofa::simulation::tree::GNode;
    root->addChild(node);

    node->addObject(solvers[0]);
    node->addObject(linear[0]);

    //merge the points of the original object to the fat added
    sofa::component::engine::MergePoints<DataTypes>* merge = new sofa::component::engine::MergePoints<DataTypes>();
    merge->f_X1.setValue(*objectMsState->getX());
    merge->f_X2.setValue(*fatmstate->getX());
    merge->init();
    merge->update();

    //update the topology merging points and tetrahedra
    BaseMeshTopology* mergedMesh = new MeshTopology();
    for (unsigned int i=0; i<merge->f_points.getValue().size(); ++i)
    {
        mergedMesh->addPoint(merge->f_points.getValue()[i][0], merge->f_points.getValue()[i][1], merge->f_points.getValue()[i][2]);
    }

    for (int i=0; i<objecTopology->getNbTetrahedra(); ++i)
    {
        mergedMesh->addTetra(objecTopology->getTetrahedron(i)[0], objecTopology->getTetrahedron(i)[1], objecTopology->getTetrahedron(i)[2], objecTopology->getTetrahedron(i)[3]);
    }
    for (int i=0; i<fatMesh->getNbTetrahedra(); ++i)
    {
        mergedMesh->addTetra((*objectMsState->getX()).size()+fatMesh->getTetrahedron(i)[0], (*objectMsState->getX()).size()+fatMesh->getTetrahedron(i)[1], (*objectMsState->getX()).size()+fatMesh->getTetrahedron(i)[2], (*objectMsState->getX()).size()+ fatMesh->getTetrahedron(i)[3]);
    }

    node->addObject(mergedMesh);

    //add mechanical state
    sofa::component::container::MechanicalObject<DataTypes>* mstate = new sofa::component::container::MechanicalObject<DataTypes>();
    node->addObject(mstate);
    mstate->init();

    FixedConstraint<DataTypes>* fc;
    behaviorNode->get(fc, sofa::core::objectmodel::BaseContext::SearchDown);
    if (fc != NULL)
    {
        behaviorNode->removeObject(fc);
        node->addObject(fc);
        fc->init();
    }

    //create collision node
    createCollisionNode(node);

    MeshLoader* meshloader;
    behaviorNode->get(meshloader, sofa::core::objectmodel::BaseContext::SearchDown);
    if (meshloader != NULL)
    {
        behaviorNode->removeObject(meshloader);
    }

    BaseMapping* mapping = new SubsetMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >(dynamic_cast<MechanicalState<Vec3Types>*>(node->getMechanicalState()), dynamic_cast<MechanicalState<Vec3Types>*>(behaviorNode->getMechanicalState()));
    behaviorNode->addObject(mapping);

    resetForce(behaviorNode,0);

    node->addChild(behaviorNode);

    //create a subsetmapping for the added fat
    sofa::simulation::tree::GNode* fatSubsetNode = createSubsetMapping(node, fatmstate, fatMesh, mass, stiffness, damping);

    //create the contact surface mapped to the object
    MechanicalState<DataTypes>* contactSurfaceObject = createContactSurfaceMapping(behaviorNode);

    //create the contact surface mapped to the fat
    MechanicalState<DataTypes>* contactSurfacefat = createContactSurfaceMapping(fatSubsetNode);

    node->init();

    //attach the object to the fat
    StiffSpringForceField<DataTypes>* spring2 = new StiffSpringForceField<DataTypes>(contactSurfaceObject, contactSurfacefat, 0, 0);

    for (unsigned int i=0; i<(*contactSurfaceObject->getX()).size(); ++i)
    {
        spring2->addSpring(i, i, 10, 0.01, 0);
    }
    node->addObject(spring2);
    spring2->init();
}

///reset residual forces
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::resetForce(core::objectmodel::BaseNode* node, int index)
{
    MechanicalState<DataTypes>* m = dynamic_cast<MechanicalState<DataTypes>*>(dynamic_cast<sofa::simulation::tree::GNode*>(node)->getMechanicalState());

    for (unsigned int i=0; i<(*m->getF()).size(); ++i)
    {
        (*m->getF())[i] = Coord(0,0,0);
    }

    index++;

    core::objectmodel::BaseNode::Children children=node->getChildren();
    for (core::objectmodel::BaseNode::Children::iterator it=children.begin(); it!= children.end(); ++it)
    {
        resetForce(*it, index);
    }
}


/// action performed at the end of the mouse interaction
template <class DataTypes>
void SculptBodyPerformer<DataTypes>::end()
{
}

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::animate()
{
    picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.body->getContext()->getMechanicalState());
        if (!mstateCollision)
        {
            std::cerr << "incompatible MState during Mouse Interaction " << std::endl;
            return;
        }

        if (!solvers.empty())
        {
            ComputeIntersectonLayer();

            //create fat mesh and mechanical state
            BaseMeshTopology* fatMesh = new MeshTopology();
            createTetraVolume(fatMesh);

            sofa::component::container::MechanicalObject<DataTypes>* fatmstate = new sofa::component::container::MechanicalObject<DataTypes>();
            fatmstate->resize(fatMesh->getNbPoints());

            for (int i=0; i<fatMesh->getNbPoints(); ++i)
            {
                DataTypes::set((*fatmstate->getX())[i], fatMesh->getPX(i), fatMesh->getPY(i), fatMesh->getPZ(i));
            }

            //merge the existing state with the added fat
            CreateNewMechanicalState(fatmstate, fatMesh);

            solvers.clear();

            startSculpt=true;
        }
    }
}

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::draw()
{

}

template <class DataTypes>
SculptBodyPerformer<DataTypes>::SculptBodyPerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i), mstateCollision(NULL), startSculpt(true)
{
    root = static_cast<simulation::Node*>(simulation::getSimulation()->getContext());
    visualRoot = static_cast<simulation::Node*>(simulation::getSimulation()->getVisualRoot()->getContext());
//           root->addObject(dynamic_cast<BaseObject*>(&eventManager));
}

#ifdef WIN32
#ifdef SOFA_DEV
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3fTypes> >  SculptBodyPerformerVec3fClass("SculptBody",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3dTypes> >  SculptBodyPerformerVec3dClass("SculptBody",true);
#endif
#endif
#endif

}
}
