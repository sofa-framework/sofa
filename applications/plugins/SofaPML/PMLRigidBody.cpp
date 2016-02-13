/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PMLRigidBody.h"

#include <PhysicalModel.h>
#include <PhysicalProperties/CellProperties.h>

#include <sofa/core/objectmodel/SPtr.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseMechanics/DiagonalMass.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/TriangleModel.h>
//#include <SofaMeshCollision/LineModel.h>
//#include <SofaMeshCollision/PointModel.h>
//using namespace sofa::component::GL;
//using namespace sofa::Core;


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::component;
using namespace sofa::component::mapping;
using namespace sofa::component::collision;
using namespace sofa::component::mass;
using namespace sofa::component::topology;

PMLRigidBody::PMLRigidBody(StructuralComponent* body, GNode * parent)
{
    parentNode = parent;
    bodyFixed = false;
    //get the parameters
    collisionsON = body->getProperties()->getBool("collision");
    name = body->getProperties()->getName();

    if(body->getProperties()->getString("mass") != "")
        initMass(body->getProperties()->getString("mass"));

    if(body->getProperties()->getString("inertiaMatrix") != "")
        initInertiaMatrix(body->getProperties()->getString("inertiaMatrix"));

    initPosition(body->getProperties()->getString("position"));
    initVelocity(body->getProperties()->getString("velocity"));
    odeSolverName = body->getProperties()->getString("odesolver");
    linearSolverName = body->getProperties()->getString("linearsolver");

    //create the structure
    createMass(body);
    createMechanicalState(body);
    createVisualModel(body);
    createCollisionModel();
    createSolver();
}


PMLRigidBody::~PMLRigidBody()
{
}


void PMLRigidBody::initMass(string m)
{
    int pos;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            SReal d=atof(s.c_str());
            massList.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
}

void PMLRigidBody::initInertiaMatrix(string m)
{
    int pos;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            SReal d=atof(s.c_str());
            inertiaMatrix.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
}

void PMLRigidBody::initPosition(string m)
{
    int pos;
    std::vector<SReal> vec;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            SReal d=atof(s.c_str());
            vec.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
    if (vec.size() >= 3)
        transPos = Vector3(vec[0],vec[1],vec[2]);
    else
        transPos = Vector3(0,0,0);

    if (vec.size() == 6)
    {
        rotPos = Quat(Vector3(1,0,0),vec[3]);
        rotPos += Quat(Vector3(0,1,0),vec[4]);
        rotPos += Quat(Vector3(0,0,1),vec[5]);
    }
    else if (vec.size() == 7)
        rotPos = Quat(vec[3], vec[4], vec[5], vec[6]);
    else
    {
        rotPos = Quat(Vector3(1,0,0),0);
        rotPos += Quat(Vector3(0,1,0),0);
        rotPos += Quat(Vector3(0,0,1),0);
    }
}

void PMLRigidBody::initVelocity(string m)
{
    int pos;
    std::vector<SReal> vec;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            SReal d=atof(s.c_str());
            vec.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
    if (vec.size() >= 3)
        transVel = Vector3(vec[0],vec[1],vec[2]);
    else
        transVel = Vector3(0,0,0);

    if (vec.size() == 6)
        rotVel += Vector3(vec[3],vec[4],vec[5]);
    else
        rotVel = Vector3(0,0,0);
}


Vector3 PMLRigidBody::getDOF(unsigned int index)
{
    return ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue()[index];
}


void PMLRigidBody::createMechanicalState(StructuralComponent* )
{
    refDOF = New<MechanicalObject<RigidTypes> >();
    refDOF->resize(1);

    //initial position and orientation of model
    ((MechanicalState<RigidTypes>*)refDOF.get())->writePositions()[0] = RigidTypes::Coord(transPos+bary, rotPos);

    //initial velocity (translation and rotation)
    ((MechanicalState<RigidTypes>*)refDOF.get())->writeVelocities()[0] = RigidTypes::Deriv(transVel, rotVel);

    parentNode->addObject(refDOF);
}


void PMLRigidBody::createTopology(StructuralComponent* body)
{

    unsigned int nbCells = body->getNumberOfCells();

    //if there is only the list of atoms in the body (no surface representation),
    //then no topology is created
    if (nbCells == 1 && body->getCell(0)->getProperties()->getType() == StructureProperties::POLY_VERTEX )
        return;

    topology = New<MeshTopology>();
    ((BaseMeshTopology*)topology.get())->clear();

    BaseMeshTopology::Triangle * tri;
    BaseMeshTopology::Quad * quad;
    Cell * pCell;
    Atom * pAtom;

    //for each pml cell, build a new Triangle or quads switch the type
    for (unsigned int cid(0) ; cid<nbCells ; cid++)
    {
        pCell = body->getCell(cid);
        switch(pCell->getProperties()->getType())
        {

        case StructureProperties::TRIANGLE :
            tri = new BaseMeshTopology::Triangle;
            for (unsigned int p(0) ; p<3 ; p++)
            {
                pAtom = (Atom*)(pCell->getStructure(p));
                (*tri)[p] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((BaseMeshTopology::SeqTriangles&)((BaseMeshTopology*)topology.get())->getTriangles()).push_back(*tri);
            break;

        case StructureProperties::QUAD :
            quad = new BaseMeshTopology::Quad;
            for (unsigned int p(0) ; p<4 ; p++)
            {
                pAtom = (Atom*)(pCell->getStructure(p));
                (*quad)[p] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology.get())->getQuads()).push_back(*quad);
            break;

        default : break;
        }
    }
}


void PMLRigidBody::createVisualModel(StructuralComponent* body)
{
    VisualNode = New<GNode>("points");
    parentNode->addChild((simulation::Node*)VisualNode.get());
    //create mechanical object
    mmodel = New<MechanicalObject<Vec3Types> >();
    //create visual model
    OglModel::SPtr vmodel = New<OglModel>();
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    SReal pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        ((MechanicalState<Vec3Types>*)mmodel.get())->writePositions()[i] = Vector3(pos[0]-bary[0],pos[1]-bary[1],pos[2]-bary[2]);
    }

    VisualNode->addObject(mmodel);

    createTopology(body);
    VisualNode->addObject(topology);

    double * color = body->getColor();
    vmodel->setColor((float)color[0], (float)color[1], (float)color[2], (float)color[3]);
    vmodel->load("","","");

    //create mappings
    mapping = New<RigidMapping<RigidTypes, Vec3Types> >();
    ((Mapping<RigidTypes, Vec3Types>*)mapping.get())->setModels((MechanicalState<RigidTypes>*)refDOF.get(), (MechanicalState<Vec3Types>*)mmodel.get());
    // BaseMapping * Vmapping = new IdentityMapping< Mapping< State<Vec3Types>, MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >
    BaseMapping::SPtr Vmapping = New<IdentityMapping< Vec3Types, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > >();
    ((Mapping< Vec3Types, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >*)Vmapping.get())->setModels((MechanicalState<Vec3Types>*)mmodel.get(), vmodel.get());

    VisualNode->addObject(mapping);
    VisualNode->addObject(Vmapping);
    VisualNode->addObject(vmodel);

}


void PMLRigidBody::createMass(StructuralComponent* body)
{
    if (! massList.empty()) // CASE MASS SPECIFIED --> Compute Inertia Matrix
    {
        StructuralComponent* atoms = body->getAtoms();
        Atom * pAtom;
        SReal masse = massList[0], totalMass=0.0;
        SReal pos[3];
        unsigned int nbPoints = atoms->getNumberOfStructures();
        SReal A,B,C,D,E,F;
        A = B = C = D = E = F = 0.0;
        bary[0] = bary[1] = bary[2] = 0.0;

        //calcul matrice d'inertie
        for (unsigned int i=0; i<nbPoints; i++)
        {
            if (massList.size() == nbPoints)
                masse = massList[i];
            pAtom = (Atom*)(atoms->getStructure(i));
            pAtom->getPosition(pos);

            // contribution of i in the inertia matrix
            A += masse * ( pos[1]*pos[1] + pos[2]*pos[2] );
            B += masse * ( pos[0]*pos[0] + pos[2]*pos[2] );
            C += masse * ( pos[0]*pos[0] + pos[1]*pos[1] );
            D += masse * pos[1] * pos[2]; //E[i]->mass*E[i]->X(2)*E[i]->X(3);
            E += masse * pos[2] * pos[0];
            F += masse * pos[0] * pos[1];
            bary[0]+=pos[0]*masse; bary[1]+=pos[1]*masse; bary[2]+=pos[2]*masse;
            totalMass += masse;
        }

        // Translate the matrix to be the inertia matrix / G
        bary[0]/=totalMass; bary[1]/=totalMass; bary[2]/=totalMass;

        A += totalMass*(bary[1]*bary[1] + bary[2]*bary[2] );
        B += totalMass*(bary[2]*bary[2] + bary[0]*bary[0] );
        C += totalMass*(bary[0]*bary[0] + bary[1]*bary[1] );
        D -= totalMass* bary[1]*bary[2];
        E -= totalMass* bary[2]*bary[0];
        F -= totalMass* bary[0]*bary[1];

        SReal coefs[9] = {A,-F, -E, -F, B, -D, -E, -D, C };

        Mat3x3d iMatrix(coefs);

        //add uniform or diagonal mass to model
        if (massList.size() ==1)
        {
            mass = New<UniformMass<Rigid3Types,Rigid3Mass> >();
            Rigid3Mass m(masse);
            m.inertiaMatrix = iMatrix;
            ((UniformMass<Rigid3Types,Rigid3Mass>*)mass.get())->setMass( m );
        }
        else
        {
            mass = New<DiagonalMass<Rigid3Types,Rigid3Mass> >();
            for (unsigned int j=0 ; j<massList.size(); j++)
            {
                Rigid3Mass m(massList[j]);
                m.inertiaMatrix = iMatrix;
                ((DiagonalMass<Rigid3Types,Rigid3Mass>*)mass.get())->addMass( m );
            }
        }
    }
    else	// CASE INERTIA MATRIX SPECIFIED
    {
        Mat3x3d iMatrix;
        switch (inertiaMatrix.size())
        {
        case 0 : //zero value --> Nothing...
            return;
        case 1 : //one value --> isotropic matrix
        {
            SReal val1 = inertiaMatrix[0];
            SReal coefs1[9] = {val1,0,0, 0,val1,0, 0,0,val1 };
            iMatrix = Mat3x3d(coefs1);
            break;
        }
        case 3 : // 3 values --> diagonal matrix
        {
            SReal coefs3[9] = {inertiaMatrix[0],0,0, 0,inertiaMatrix[1],0, 0,0,inertiaMatrix[2] };
            iMatrix = Mat3x3d(coefs3);
            break;
        }
        case 6 : // 6 values --> symetric matrix
        {
            SReal coefs9[9] = {inertiaMatrix[0],inertiaMatrix[1],inertiaMatrix[2], \
                    inertiaMatrix[1],inertiaMatrix[3],inertiaMatrix[4], \
                    inertiaMatrix[2],inertiaMatrix[4],inertiaMatrix[5]
                              };
            iMatrix = Mat3x3d(coefs9);
            break;
        }
        default : // else --> houston, we've got a problem!
            cerr<<"WARNING building "<<name<<" object : inertia matrix not properly defined."<<endl;
            return;
        }
        mass = New<UniformMass<Rigid3Types,Rigid3Mass> >();
        Rigid3Mass m(1.0);
        m.inertiaMatrix = iMatrix;
        ((UniformMass<Rigid3Types,Rigid3Mass>*)mass.get())->setMass( m );
    }

    if (mass)
        parentNode->addObject(mass);
}

void PMLRigidBody::createCollisionModel()
{
    if (collisionsON)
    {
        //CollisionNode = new GNode("Collision");
        //parentNode->addChild(CollisionNode);

        /*CollisionNode->addObject(mmodel);
        CollisionNode->addObject(topology);
        CollisionNode->addObject(mapping);*/

        TriangleModel::SPtr cmodel = New<TriangleModel>();
        //LineModel *lmodel = new LineModel;
        //PointModel *pmodel = new PointModel;
        VisualNode->addObject(cmodel);
        //VisualNode->addObject(lmodel);
        //VisualNode->addObject(pmodel);

        cmodel->init();
        //lmodel->init();
        //pmodel->init();
    }
}


bool PMLRigidBody::FusionBody(PMLBody* )
{
    return false;
}


}
}
}
