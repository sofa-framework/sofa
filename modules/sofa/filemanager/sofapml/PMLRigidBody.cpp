/***************************************************************************
								PMLRigidBody
                             -------------------
    begin             : August 18th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2006/02/25 13:51:44 $
    Version           : $Revision: 0.2 $
 ***************************************************************************/

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
#include <CellProperties.h>

#include "sofa/defaulttype/Vec3Types.h"
#include "sofa/component/mapping/RigidMapping.h"
#include "sofa/component/mapping/IdentityMapping.h"
//#include "sofa/componentCore/MappedModel.h"
#include "sofa/component/mass/UniformMass.h"
#include "sofa/component/mass/DiagonalMass.h"
#include "sofa/component/topology/MeshTopology.h"
#include "sofa/component/collision/TriangleModel.h"
#include "sofa/component/collision/LineModel.h"
#include "sofa/component/collision/PointModel.h"
//using namespace sofa::component::GL;
using namespace sofa::component;
using namespace sofa::component::mapping;
using namespace sofa::component::collision;
using namespace sofa::component::mass;
using namespace sofa::component::topology;
//using namespace sofa::Core;


namespace sofa
{

namespace filemanager
{

namespace pml
{

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

    //create the structure
    createMass(body);
    createMechanicalState(body);
    createVisualModel(body);
    createCollisionModel();
}


PMLRigidBody::~PMLRigidBody()
{
    if(mmodel) delete mmodel;
    if (mapping) delete mapping;
    if (VisualNode) delete VisualNode;
    if (CollisionNode) delete CollisionNode;
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
            double d=atof(s.c_str());
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
            double d=atof(s.c_str());
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
    std::vector<double> vec;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            double d=atof(s.c_str());
            vec.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
    if (vec.size() >= 3)
        transPos = Vec3d(vec[0],vec[1],vec[2]);
    else
        transPos = Vec3d(0,0,0);

    if (vec.size() == 6)
    {
        rotPos = Quat(Vec3d(1,0,0),vec[3]);
        rotPos += Quat(Vec3d(0,1,0),vec[4]);
        rotPos += Quat(Vec3d(0,0,1),vec[5]);
    }
    else if (vec.size() == 7)
        rotPos = Quat(vec[3], vec[4], vec[5], vec[6]);
    else
    {
        rotPos = Quat(Vec3d(1,0,0),0);
        rotPos += Quat(Vec3d(0,1,0),0);
        rotPos += Quat(Vec3d(0,0,1),0);
    }
}

void PMLRigidBody::initVelocity(string m)
{
    int pos;
    std::vector<double> vec;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            double d=atof(s.c_str());
            vec.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
    if (vec.size() >= 3)
        transVel = Vec3d(vec[0],vec[1],vec[2]);
    else
        transVel = Vec3d(0,0,0);

    if (vec.size() == 6)
        rotVel += Vec3d(vec[3],vec[4],vec[5]);
    else
        rotVel = Vec3d(0,0,0);
}


Vec3d PMLRigidBody::getDOF(unsigned int index)
{
    return (*((MechanicalState<Vec3dTypes>*)mmodel)->getX())[index];
}


void PMLRigidBody::createMechanicalState(StructuralComponent* )
{
    refDOF = new MechanicalObject<RigidTypes>;
    refDOF->resize(1);

    //initial position and orientation of model
    (*((MechanicalState<RigidTypes>*)refDOF)->getX())[0] = RigidTypes::Coord(transPos+bary, rotPos);

    //initial velocity (translation and rotation)
    (*((MechanicalState<RigidTypes>*)refDOF)->getV())[0] = RigidTypes::Deriv(transVel, rotVel);

    parentNode->addObject(refDOF);
}


void PMLRigidBody::createTopology(StructuralComponent* body)
{

    unsigned int nbCells = body->getNumberOfCells();

    //if there is only the list of atoms in the body (no surface representation),
    //then no topology is created
    if (nbCells == 1 && body->getCell(0)->getProperties()->getType() == StructureProperties::POLY_VERTEX )
        return;

    topology = new MeshTopology();
    ((MeshTopology*)topology)->clear();

    MeshTopology::Triangle * tri;
    MeshTopology::Quad * quad;
    Cell * pCell;
    Atom * pAtom;

    //for each pml cell, build a new Triangle or quads switch the type
    for (unsigned int cid(0) ; cid<nbCells ; cid++)
    {
        pCell = body->getCell(cid);
        switch(pCell->getProperties()->getType())
        {

        case StructureProperties::TRIANGLE :
            tri = new MeshTopology::Triangle;
            for (unsigned int p(0) ; p<3 ; p++)
            {
                pAtom = (Atom*)(pCell->getStructure(p));
                (*tri)[p] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((MeshTopology::SeqTriangles&)((MeshTopology*)topology)->getTriangles()).push_back(*tri);
            break;

        case StructureProperties::QUAD :
            quad = new MeshTopology::Quad;
            for (unsigned int p(0) ; p<4 ; p++)
            {
                pAtom = (Atom*)(pCell->getStructure(p));
                (*quad)[p] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((MeshTopology::SeqQuads&)((MeshTopology*)topology)->getQuads()).push_back(*quad);
            break;

        default : break;
        }
    }
}


void PMLRigidBody::createVisualModel(StructuralComponent* body)
{
    VisualNode = new GNode("points");
    parentNode->addChild(VisualNode);
    //create mechanical object
    mmodel = new MechanicalObject<Vec3dTypes>;
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    double pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        (*((MechanicalState<Vec3dTypes>*)mmodel)->getX())[i] = Vec3d(pos[0]-bary[0],pos[1]-bary[1],pos[2]-bary[2]);
    }

    VisualNode->addObject(mmodel);

    createTopology(body);
    VisualNode->addObject(topology);

    //create visual model
    OglModel * vmodel = new OglModel;

    double * color = body->getColor();
    vmodel->setColor((float)color[0], (float)color[1], (float)color[2], (float)color[3]);
    vmodel->load("","","");

    //create mappings
    mapping = new RigidMapping< MechanicalMapping<MechanicalState<RigidTypes>, MechanicalState<Vec3dTypes> > >( (MechanicalState<RigidTypes>*)refDOF, (MechanicalState<Vec3dTypes>*)mmodel);
    BaseMapping * Vmapping = new IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >((MechanicalState<Vec3dTypes>*)mmodel, vmodel);

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
        double masse = massList[0], totalMass=0.0;
        double pos[3];
        unsigned int nbPoints = atoms->getNumberOfStructures();
        double A,B,C,D,E,F;
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

        double coefs[9] = {A,-F, -E, -F, B, -D, -E, -D, C };

        Mat3x3d iMatrix(coefs);

        //add uniform or diagonal mass to model
        if (massList.size() ==1)
        {
            mass = new UniformMass<RigidTypes,RigidMass>;
            RigidMass m(masse);
            m.inertiaMatrix = iMatrix;
            ((UniformMass<RigidTypes,RigidMass>*)mass)->setMass( m );
        }
        else
        {
            mass = new DiagonalMass<RigidTypes,RigidMass>;
            for (unsigned int j=0 ; j<massList.size(); j++)
            {
                RigidMass m(massList[j]);
                m.inertiaMatrix = iMatrix;
                ((DiagonalMass<RigidTypes,RigidMass>*)mass)->addMass( m );
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
            double val1 = inertiaMatrix[0];
            double coefs1[9] = {val1,0,0, 0,val1,0, 0,0,val1 };
            iMatrix = Mat3x3d(coefs1);
            break;
        }
        case 3 : // 3 values --> diagonal matrix
        {
            double coefs3[9] = {inertiaMatrix[0],0,0, 0,inertiaMatrix[1],0, 0,0,inertiaMatrix[2] };
            iMatrix = Mat3x3d(coefs3);
            break;
        }
        case 6 : // 6 values --> symetric matrix
        {
            double coefs9[9] = {inertiaMatrix[0],inertiaMatrix[1],inertiaMatrix[2], \
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
        mass = new UniformMass<RigidTypes,RigidMass>;
        RigidMass m(1.0);
        m.inertiaMatrix = iMatrix;
        ((UniformMass<RigidTypes,RigidMass>*)mass)->setMass( m );
    }

    if (mass)
        parentNode->addObject(mass);
}

void PMLRigidBody::createCollisionModel()
{
    if (collisionsON)
    {
        CollisionNode = new GNode("Collision");
        parentNode->addChild(CollisionNode);

        CollisionNode->addObject(mmodel);
        CollisionNode->addObject(topology);
        CollisionNode->addObject(mapping);

        TriangleModel * cmodel = new TriangleModel;
        LineModel *lmodel = new LineModel;
        PointModel *pmodel = new PointModel;
        CollisionNode->addObject(cmodel);
        CollisionNode->addObject(lmodel);
        CollisionNode->addObject(pmodel);

        cmodel->init();
        lmodel->init();
        pmodel->init();
    }
}


bool PMLRigidBody::FusionBody(PMLBody* )
{
    return false;
}


}
}
}
