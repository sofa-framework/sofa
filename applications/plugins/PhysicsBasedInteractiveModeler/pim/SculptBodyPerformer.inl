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

#include "SculptBodyPerformer.h"
#include <sofa/component/collision/CubeModel.h>
#include <sofa/simulation/common/Simulation.h>
#include "PointsOnSurface.h"
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/component/forcefield/TriangularFEMForceField.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/engine/MergePoints.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/BaseMapping.h>

namespace plugins
{

namespace pim
{

using namespace sofa::component::mapping;
using namespace sofa::component::engine;
using namespace sofa::component::forcefield;
using namespace sofa::component::container;
using namespace sofa::component::visualmodel;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::core;

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::computeNeighborhood()
{
    sofa::core::componentmodel::topology::BaseMeshTopology::VerticesAroundVertex vav;
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

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::execute()
{
    picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        mstateCollision = dynamic_cast< core::componentmodel::behavior::MechanicalState<DataTypes>*  >(picked.body->getContext()->getMechanicalState());

        if (!mstateCollision)
        {
            std::cerr << "uncompatible MState during Mouse Interaction " << std::endl;
            return;
        }
        typename DataTypes::VecCoord& X = *mstateCollision->getX();
        const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = picked.body->getMeshTopology()->getTriangle(picked.indexCollisionElement);

        // gets the diagonal of the bounding box
        CubeModel* cModel = dynamic_cast<CubeModel*>(picked.body->getFirst());
        Cube bbox = Cube(cModel,0);
        typename DataTypes::Real diagonal = (typename DataTypes::Real)((bbox.minVect() - bbox.maxVect()).norm()/4); //division by an integer, is it desired?
        diagonal *= (typename DataTypes::Real) (scale/99); //idem here?
        typename DataTypes::Real dist, W;

        // gets the nearest point from the picked collision element (Triangle)
        unsigned int nearest_point = elem[0];
        if ((X[nearest_point] - picked.point).norm() > (X[elem[1]] - picked.point).norm())
        {
            nearest_point = elem[1];
        }
        if ((X[nearest_point] - picked.point).norm() > (X[elem[2]] - picked.point).norm())
        {
            nearest_point = elem[2];
        }

        alreadyCheckedVertex.clear();
        vertexNeighborhood.insert(nearest_point);

        do
        {
            vertexInInfluenceZone.clear();
            for( std::set<unsigned int>::const_iterator iter = vertexNeighborhood.begin(); iter != vertexNeighborhood.end(); ++iter )
            {
                if (fixedPoints.find(*iter) == fixedPoints.end())
                {
                    dist = (X[*iter]-picked.point).norm();
                    if (dist < diagonal)
                    {
                        vertexInInfluenceZone.insert(*iter);

                        const sofa::core::componentmodel::topology::BaseMeshTopology::TrianglesAroundVertex& tav = picked.body->getMeshTopology()->getTrianglesAroundVertex(*iter);

                        drawFacets.insert(tav.begin(), tav.end());

                        if (!checkedFix)
                        {
                            Coord normal;

                            // computes the normal of the point
                            for (unsigned int i=0; i<tav.size(); ++i)
                            {
                                const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& t = picked.body->getMeshTopology()->getTriangle(tav[i]);
                                const Coord p0 = X[t[0]];
                                const Coord p1 = X[t[1]];
                                const Coord p2 = X[t[2]];
                                normal += cross(p1-p0, p2-p0);
                            }
                            normal.normalize();

                            // Wendland kernel
                            W = pow(pow(1-dist/diagonal,2),2)*(4*dist/diagonal+1);
                            // computes new position
                            X[*iter] += normal*W*force;
                            if (force > 0.000000001)
                                modifiedVertex.insert(*iter);
                            if (force > 0.001)
                                modified.insert(*iter);
                        }
                    }
                }

                alreadyCheckedVertex.insert(*iter);
            }
            computeNeighborhood();
        }
        while(!vertexInInfluenceZone.empty());
        sofa::core::BaseMapping *mapping;
        mstateCollision->getContext()->get(mapping); assert(mapping);
        if(mapping)
            mapping->init();
        if (checkedFix)
        {
            fixedPoints.insert(alreadyCheckedVertex.begin(), alreadyCheckedVertex.end());
            fixedFacets.insert(drawFacets.begin(), drawFacets.end());
        }
    }
}


template <class DataTypes>
void SculptBodyPerformer<DataTypes>::end()
{
    if (checkedFix) return;
    picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        typename DataTypes::VecCoord& X = *mstateCollision->getX();
        typename DataTypes::VecCoord& X0 = *mstateCollision->getX0();

        fatMesh->clear();
        seqPoints.clear();
        triangleChecked.clear();
        vertexIndex = 0;
        vertexMap.clear();

        for( std::set<unsigned int>::const_iterator iter = modifiedVertex.begin(); iter != modifiedVertex.end(); ++iter )
        {
            fatMesh->addPoint(X[*iter][0], X[*iter][1], X[*iter][2]);
            seqPoints.push_back(defaulttype::Vec<3,SReal>(X[*iter][0], X[*iter][1], X[*iter][2]));
            vertexMap.insert(std::make_pair(*iter, vertexIndex));
            vertexIndex++;

            fatMesh->addPoint(X0[*iter][0], X0[*iter][1], X0[*iter][2]);
            seqPoints.push_back(defaulttype::Vec<3,SReal>(X0[*iter][0], X0[*iter][1], X0[*iter][2]));
            vertexMap.insert(std::make_pair(*iter, vertexIndex));
            vertexIndex++;

            surfacePoint.push_back(X0[*iter]);

            const sofa::core::componentmodel::topology::BaseMeshTopology::VerticesAroundVertex& vav = picked.body->getMeshTopology()->getVerticesAroundVertex(*iter);

            for (unsigned int i=0; i<vav.size(); ++i)
            {
                if (modifiedVertex.find(vav[i]) == modifiedVertex.end())
                {
                    fatMesh->addPoint(X[vav[i]][0], X[vav[i]][1], X[vav[i]][2]);
                    seqPoints.push_back(defaulttype::Vec<3,SReal>(X[vav[i]][0], X[vav[i]][1], X[vav[i]][2]));
                    vertexMap.insert(std::make_pair(vav[i], vertexIndex));
                    vertexMap.insert(std::make_pair(vav[i], vertexIndex));
                    vertexIndex++;

                    surfacePoint.push_back(X[vav[i]]);
                }
            }
        }

        for( std::set<unsigned int>::const_iterator iter = modifiedVertex.begin(); iter != modifiedVertex.end(); ++iter )
        {
            const sofa::core::componentmodel::topology::BaseMeshTopology::TrianglesAroundVertex& tav = picked.body->getMeshTopology()->getTrianglesAroundVertex(*iter);

            for (unsigned int i=0; i<tav.size(); ++i)
            {
                if (triangleChecked.insert(tav[i]).second == true)
                {
                    const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& t = picked.body->getMeshTopology()->getTriangle(tav[i]);
                    std::multimap<unsigned int, unsigned int>::const_iterator it1 = vertexMap.find(t[0]);
                    std::multimap<unsigned int, unsigned int>::const_iterator it2 = vertexMap.find(t[1]);
                    std::multimap<unsigned int, unsigned int>::const_iterator it3 = vertexMap.find(t[2]);

                    fatMesh->addTriangle(it1->second, it2->second, it3->second);
                    it1++; it2++; it3++;
                    fatMesh->addTriangle(it1->second, it3->second, it2->second);
                }
            }
        }

        if (meshStuffed!=NULL)
        {
            delete (meshStuffed);
        }
        meshStuffed = new misc::MeshTetraStuffing();
        meshStuffed->inputPoints.setValue(seqPoints);
        meshStuffed->inputTriangles.setValue(fatMesh->getTriangles());
        meshStuffed->bSnapPoints.setValue(true);
        meshStuffed->bSplitTetrahedra.setValue(true);
        meshStuffed->bDraw.setValue(true);
        meshStuffed->size.setValue(0.05);
        meshStuffed->alphaLong.setValue(0.3);
        meshStuffed->alphaShort.setValue(0.4);
        meshStuffed->init();
    }
}

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::animate(bool checked)
{
    if (checkedFix) return;
    if (checked)
    {

/////////////////// Fill matter topology //////////////////////////
        dynamicMatterNode = root->getChild("DynamicMatter");
        matterMesh = dynamicMatterNode->getMeshTopology();
        matterMesh->clear();
        for (unsigned int i=0; i<meshStuffed->outputPoints.getValue().size(); ++i)
        {
            matterMesh->addPoint(meshStuffed->outputPoints.getValue()[i][0], meshStuffed->outputPoints.getValue()[i][1],
                    meshStuffed->outputPoints.getValue()[i][2]);
        }
        for (unsigned int i=0; i<meshStuffed->outputTetrahedra.getValue().size(); ++i)
        {
            matterMesh->addTetra(meshStuffed->outputTetrahedra.getValue()[i][0], meshStuffed->outputTetrahedra.getValue()[i][1],
                    meshStuffed->outputTetrahedra.getValue()[i][2], meshStuffed->outputTetrahedra.getValue()[i][3]);
        }
//////////////////////////////////////////////////////////////////////

/////////////////// Fill matter mstate //////////////////////////
        matterMstate = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(dynamicMatterNode->getMechanicalState());
        matterMstate->resize(0);
        sculptedPointsNode2 = dynamicMatterNode->getChild("SculptedPoints");
        dynamicMatterNode->removeChild(sculptedPointsNode2);
        dynamicMatterNode->init();

////////////////////////body//////////////////////////////////////
        typename DataTypes::VecCoord& X = *mstateCollision->getX();
        bodyNode = root->getChild("Body");
        sculptedPointsNode = bodyNode->getChild("SculptedPoints");
        MechanicalState<DataTypes>* modif = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(sculptedPointsNode->getMechanicalState());
        typename DataTypes::VecCoord& modifX = *modif->getX();
        typename DataTypes::VecCoord& modifX0 = *modif->getX0();

        plugins::pim::PointsOnSurface<DataTypes>* pos2 = new plugins::pim::PointsOnSurface<DataTypes>();
        pos2->radius.setValue(0.06);
        pos2->f_surfacePoints.setValue(*matterMstate->getX());
        pos2->f_setOfPoints.setValue(*mstateCollision->getX());
        pos2->init();
        pos2->update();

        modif->resize(pos2->f_outputPoints.getValue().size());
        for (unsigned int i=0; i<pos2->f_outputPoints.getValue().size(); ++i)
        {
            modifX[i] = X[pos2->f_outputPoints.getValue()[i]];
            modifX0[i] = X[pos2->f_outputPoints.getValue()[i]];
        }

        BaseMapping * subSetMapping;
        subSetMapping = new SubsetMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >((MechanicalState<Vec3Types>*)mstateCollision, (MechanicalState<Vec3Types>*)modif);
        sculptedPointsNode->addObject(subSetMapping);
        subSetMapping->init();
///////////////////////////////////////////////////////////////////////////

/////Bary
        BaseMapping * barycentricMapping;
        MechanicalState<DataTypes>* mm = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(sculptedPointsNode2->getMechanicalState());
        mm->resize(modifX.size());
        typename DataTypes::VecCoord& mmX = *mm->getX();
        typename DataTypes::VecCoord& mmX0 = *mm->getX0();

        for (unsigned int i=0; i<modifX.size(); ++i)
        {
            mmX[i] = modifX[i];
            mmX0[i] = modifX0[i];
        }

        barycentricMapping = new BarycentricMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >((MechanicalState<Vec3Types>*)matterMstate, (MechanicalState<Vec3Types>*)mm);
        sculptedPointsNode2->addObject(barycentricMapping);
////////
        dynamicMatterNode->addChild(sculptedPointsNode2);
        sculptedPointsNode2->init();
////////////////////////////////////////////////////////////////

//////////////////////////set static points//////////////////////////
        plugins::pim::PointsOnSurface<DataTypes>* pos = new plugins::pim::PointsOnSurface<DataTypes>();
        pos->f_surfacePoints.setValue(surfacePoint);
        pos->f_setOfPoints.setValue(*matterMstate->getX());
        pos->init();
        pos->update();

        staticMatterNode = root->getChild("StaticMatter");
        MechanicalState<DataTypes>* SubsetMstate = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(staticMatterNode->getMechanicalState());
        typename DataTypes::VecCoord& S = *SubsetMstate->getX();
        S.resize(pos->f_outputPoints.getValue().size());
        typename DataTypes::VecCoord& ST = *matterMstate->getX();
        for (unsigned int i=0; i<pos->f_outputPoints.getValue().size(); ++i)
            S[i] = ST[pos->f_outputPoints.getValue()[i]];

        typename DataTypes::VecCoord& S0 = *SubsetMstate->getX0();
        S0 = S;
        staticMatterNode->init();
//////////////////////////////////////////////////////////////////////

/////////////////fix statics to dynamics//////////////////////////////////
        StiffSpringForceField<DataTypes>* spring = dynamic_cast<StiffSpringForceField<DataTypes>*>(root->getObject("fixSpring"));
        for(unsigned int i=0; i<S.size(); ++i)
            spring->addSpring(i, pos->f_outputPoints.getValue()[i],spring->getStiffness(),spring->getDamping(),0.0);
//////////////////////////////////////////////////////////////////////////

/////////////////fix body to dynamic matter//////////////////////////////////
        StiffSpringForceField<DataTypes>* spring2 = dynamic_cast<StiffSpringForceField<DataTypes>*>(root->getObject("fix2Spring"));
        for(unsigned int i=0; i<mm->getX()->size(); ++i)
            spring2->addSpring(i, i,spring->getStiffness(),spring->getDamping(),0.0);
//////////////////////////////////////////////////////////////////////////

    }
    else
    {
//              root->removeChild(addedMateriaNode);
//              root->removeChild(matterNode);
    }
}

template <class DataTypes>
void SculptBodyPerformer<DataTypes>::draw()
{
    if (picked.body == NULL) return;
    if (mstateCollision == NULL) return;
    typename DataTypes::VecCoord& X = *mstateCollision->getX();
    core::componentmodel::topology::BaseMeshTopology* topo = picked.body->getMeshTopology();

    std::vector< Vector3 > points;
    std::vector< Vec<3,int> > indices;
    std::vector< Vector3 > normals;

    int index=0;
    for( std::set<unsigned int>::const_iterator iter = drawFacets.begin(); iter != drawFacets.end(); ++iter )
    {
        const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = topo->getTriangle(*iter);
        if (alreadyCheckedVertex.find(elem[0]) != alreadyCheckedVertex.end() &&
            alreadyCheckedVertex.find(elem[1]) != alreadyCheckedVertex.end() &&
            alreadyCheckedVertex.find(elem[2]) != alreadyCheckedVertex.end())
        {
            TriangleModel* tModel = dynamic_cast<TriangleModel*>(picked.body);
            collision::Triangle t(tModel,*iter);
            normals.push_back(t.n());
            points.push_back(X[elem[0]]);
            points.push_back(X[elem[1]]);
            points.push_back(X[elem[2]]);
            indices.push_back(Vec<3,int>(index,index+1,index+2));
            index+=3;
        }
    }
    sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(true);
    simulation::getSimulation()->DrawUtility.drawTriangles(points, indices, normals, Vec<4,float>(1,0,1,0.5));
    sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(false);


//           for (unsigned int i=0; i < fatMesh->getNbTriangles(); ++i)
//           {
//               const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = fatMesh->getTriangle(i);
//
// 		  Vector3 C(fatMesh->getPX(elem[0]), fatMesh->getPY(elem[0]), fatMesh->getPZ(elem[0]));
// 		  Vector3 D(fatMesh->getPX(elem[1]), fatMesh->getPY(elem[1]), fatMesh->getPZ(elem[1]));
// 		  Vector3 E(fatMesh->getPX(elem[2]), fatMesh->getPY(elem[2]), fatMesh->getPZ(elem[2]));
//
// 		  Vector3 A = D - C;
// 		  Vector3 B = E - D;
// 		  Vector3 n = A.cross(B);
//
// 		  n.normalize();
//
// 		  normals.push_back(n);
//
// 		  points.push_back(C);
// 		  points.push_back(D);
// 		  points.push_back(E);
// 		  indices.push_back(Vec<3,int>(index,index+1,index+2));
// 		  index+=3;
//           }
//                   sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(true);
//                   simulation::getSimulation()->DrawUtility.drawTriangles(points, indices, normals, Vec<4,float>(1,0,1,0.5));
//                   sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(false);
//
//
//
//           index=0;
// 	  points.clear();
//           for (int i=0; i < fatMesh->getNbPoints(); ++i)
//           {
//
// 		  Vector3 A(fatMesh->getPX(i), fatMesh->getPY(i), fatMesh->getPZ(i));
// 		  points.push_back(A);
//           }
//
//                   sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(true);
//                   simulation::getSimulation()->DrawUtility.drawPoints(points, 5, Vec<4,float>(1,0,0,1));
//                   sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(false);

    /*          glPointSize(5);
              glColor4f(1,0,0,1);
              glBegin(GL_POINTS);
              for( std::set<unsigned int>::const_iterator iter = fixedPoints.begin(); iter != fixedPoints.end(); ++iter )
              {
                  glVertex3d(X[*iter][0],X[*iter][1],X[*iter][2]);
              }
              glEnd();*/
    points.clear();
    indices.clear();
    normals.clear();
    index=0;
    for( std::set<unsigned int>::const_iterator iter = fixedFacets.begin(); iter != fixedFacets.end(); ++iter )
    {
        const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& elem = topo->getTriangle(*iter);
        if (fixedPoints.find(elem[0]) != fixedPoints.end() &&
            fixedPoints.find(elem[1]) != fixedPoints.end() &&
            fixedPoints.find(elem[2]) != fixedPoints.end())
        {
            TriangleModel* tModel = dynamic_cast<TriangleModel*>(picked.body);
            collision::Triangle t(tModel,*iter);
            normals.push_back(t.n());
            points.push_back(X[elem[0]]);
            points.push_back(X[elem[1]]);
            points.push_back(X[elem[2]]);
            indices.push_back(Vec<3,int>(index,index+1,index+2));
            index+=3;
        }
    }
    sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(true);
    simulation::getSimulation()->DrawUtility.drawTriangles(points, indices, normals, Vec<4,float>(0,0.5,0,1));
    sofa::simulation::getSimulation()->DrawUtility.setLightingEnabled(false);


    /*          glPointSize(5);
              glColor4f(1,0,0,1);
              glBegin(GL_POINTS);
              for( std::set<unsigned int>::const_iterator iter = modifiedVertex.begin(); iter != modifiedVertex.end(); ++iter )
              {
                  glVertex3d(X[*iter][0],X[*iter][1],X[*iter][2]);
              }
              glEnd();*/
}

template <class DataTypes>
SculptBodyPerformer<DataTypes>::SculptBodyPerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i), mstateCollision(NULL), vertexIndex(0), meshStuffed(NULL), matterMesh(NULL), matterMstate(NULL), matterFem(NULL), matterMass(NULL)
{
    root = static_cast<simulation::Node*>(simulation::getSimulation()->getContext());
    fatMesh = new sofa::component::topology::MeshTopology();
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
