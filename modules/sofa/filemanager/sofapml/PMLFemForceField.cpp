/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include "PMLFemForceField.h"

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <CellProperties.h>
#include <PMLTransform.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::mass;
using namespace sofa::component::mapping;
using namespace sofa::component::forcefield;
using namespace sofa::component::topology;
using namespace sofa::component;

PMLFemForceField::PMLFemForceField(StructuralComponent* body, GNode * parent)
{
    parentNode = parent;
    //get the parameters
    collisionsON = body->getProperties()->getBool("collision");
    name = body->getProperties()->getName();

    if(body->getProperties()->getString("mass") != "")
        initMass(body->getProperties()->getString("mass"));

    if(body->getProperties()->getString("density") != "")
        initDensity(body->getProperties()->getString("density"));

    young = body->getProperties()->getDouble("young");
    poisson = body->getProperties()->getDouble("poisson");
    deformationType = body->getProperties()->getString("deformation");
    odeSolverName = body->getProperties()->getString("odesolver");
    linearSolverName = body->getProperties()->getString("linearsolver");

    //create the structure
    createMechanicalState(body);
    createTopology(body);
    createMass(body);
    createVisualModel(body);
    createForceField();
    createCollisionModel();
    createSolver();
}


PMLFemForceField::~PMLFemForceField()
{
    if(mmodel) delete mmodel;
    //if(pmodel) delete pmodel;
    //if(lmodel) delete lmodel;
    if(tmodel) delete tmodel;
}


//read the mass parameter
void PMLFemForceField::initMass(string m)
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

void PMLFemForceField::initDensity(string m)
{
    int pos;
    while(!m.empty())
    {
        pos = m.find(' ', 0);
        if(pos != 0)
        {
            string s=m.substr(0,pos);
            SReal d=atof(s.c_str());
            density.push_back(d);
            m.erase(0,pos);
        }
        else
            m.erase(0,1);
    }
}


//convenient method to tesselate a hexahedron to 5 tetrahedra
// used by createTopology method
BaseMeshTopology::Tetra * PMLFemForceField::Tesselate(Cell* pCell)
{
    BaseMeshTopology::Tetra *tet = new BaseMeshTopology::Tetra[5];
    Atom *pAtom;
    int index[8];

    for (int i(0) ; i<8 ; i++)
    {
        pAtom = (Atom*)(pCell->getStructure(i));
        index[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
    }

    tet[0][0]=index[0]; tet[0][1]=index[1]; tet[0][2]=index[4]; tet[0][3]=index[3];
    tet[1][0]=index[2]; tet[1][1]=index[3]; tet[1][2]=index[6]; tet[1][3]=index[1];
    tet[2][0]=index[5]; tet[2][1]=index[4]; tet[2][2]=index[1]; tet[2][3]=index[6];
    tet[3][0]=index[7]; tet[3][1]=index[6]; tet[3][2]=index[3]; tet[3][3]=index[4];
    tet[4][0]=index[1]; tet[4][1]=index[6]; tet[4][2]=index[4]; tet[4][3]=index[3];

    return tet;
}

Vector3 PMLFemForceField::getDOF(unsigned int index)
{
    return (*((MechanicalState<Vec3Types>*)mmodel)->getX())[index];
}

//creation of the mechanical model
//each pml atom constituing the body correspond to a DOF
void PMLFemForceField::createMechanicalState(StructuralComponent* body)
{
    mmodel = new MechanicalObject<Vec3Types>;
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    SReal pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        (*((MechanicalState<Vec3Types>*)mmodel)->getX())[i] = Vector3(pos[0],pos[1],pos[2]);
    }

    parentNode->addObject(mmodel);

}


// creation of the topology
// topology constituted exclusively by tetrahedra
// --> if there is hexahedrons, they are tesselated in 5 tetrahedra
void PMLFemForceField::createTopology(StructuralComponent* body)
{
    topology = new MeshTopology();
    ((BaseMeshTopology*)topology)->clear();

    unsigned int nbCells = body->getNumberOfCells();
    BaseMeshTopology::Tetra * tet;
    BaseMeshTopology::Line * line;
    Cell * pCell;
    Atom * pAtom;

    //for each pml cell, build 1 or 5 tetrahedra
    for (unsigned int cid(0) ; cid<nbCells ; cid++)
    {
        pCell = body->getCell(cid);
        switch(pCell->getProperties()->getType())
        {

        case StructureProperties::HEXAHEDRON :
            tet = Tesselate(pCell);
            for (unsigned int p(0) ; p<5 ; p++)
            {
                ((BaseMeshTopology::SeqTetrahedra&)((BaseMeshTopology*)topology)->getTetrahedra()).push_back(tet[p]);
                for (unsigned int l1=0 ; l1<4 ; l1++)
                {
                    for (unsigned int l2=l1+1 ; l2<4 ; l2++)
                    {
                        line = new BaseMeshTopology::Line;
                        (*line)[0] = tet[p][l1];
                        (*line)[1] = tet[p][l2];
                        ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology)->getLines()).push_back(*line);
                    }
                }
            }
            break;

        case StructureProperties::TETRAHEDRON :
            tet = new BaseMeshTopology::Tetra;
            for (unsigned int i(0) ; i<4 ; i++)
            {
                pAtom = (Atom*)(pCell->getStructure(i));
                (*tet)[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
                for (unsigned int l1=0 ; l1<4 ; l1++)
                {
                    for (unsigned int l2=l1+1 ; l2<4 ; l2++)
                    {
                        line = new BaseMeshTopology::Line;
                        (*line)[0] = (*tet)[l1];
                        (*line)[1] = (*tet)[l2];
                        ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology)->getLines()).push_back(*line);
                    }
                }
            }
            ((BaseMeshTopology::SeqTetrahedra&)((BaseMeshTopology*)topology)->getTetrahedra()).push_back(*tet);
            break;

        default : break;

        }
    }
    parentNode->addObject(topology);
}


//creation of the mass
//normaly there 1 value OR nbDOFs values (OR 0 if not specified)
void PMLFemForceField::createMass(StructuralComponent* body)
{
    //if no mass specified...
    if (massList.size() == 0)
    {
        //...normally density is!
        if (density.size() != 0)
        {
            //BUILDING WITH DENSITY PROPERTY
            if (density.size() > 1 && density.size() != ((MechanicalState<Vec3Types>*)mmodel)->getX()->size())
            {
                cerr<<"WARNING building "<<name<<" object : density property not properly defined."<<endl;
                return;
            }
            else
            {
                //init the mass list
                for (unsigned int i=0 ; i<((MechanicalState<Vec3Types>*)mmodel)->getX()->size() ; i++)
                    massList.push_back(0.0);

                SReal m;
                Cell * pCell;
                Atom * pAtom;

                //for each atom of each cell...
                for (unsigned int cid(0) ; cid<body->getNumberOfCells(); cid++)
                {
                    pCell = body->getCell(cid);
                    SReal volumeCell = pCell->volume();
                    for (unsigned int j(0) ; j< pCell->getNumberOfStructures() ; j++)
                    {
                        pAtom = (Atom*)(pCell->getStructure(j));
                        SReal dens = density.size()>1?density[AtomsToDOFsIndexes[pAtom->getIndex()]]:density[0];
                        //mass of atom += atom density * cell volume / nb atoms in cell
                        m = dens * volumeCell / pCell->getNumberOfStructures();
                        massList[AtomsToDOFsIndexes[pAtom->getIndex()]] += m;
                    }
                }

                mass = new DiagonalMass<Vec3Types,SReal>;
                for (unsigned int im=0 ; im<massList.size() ; im++)
                {
                    ((DiagonalMass<Vec3Types,SReal>*)mass)->addMass( massList[im] );
                }
            }
        }
    } //BUILDING WITH MASS PROPERTY
    else
    {
        //if there is 1 value --> uniform mass for all the model
        if (massList.size() == 1)
        {
            mass = new UniformMass<Vec3Types,SReal>;
            ((UniformMass<Vec3Types,SReal>*)mass)->setMass( massList[0] );
        }
        else
        {
            //if there nbDofs values --> diagonal mass (one value for each dof)
            if (massList.size() == ((MechanicalState<Vec3Types>*)mmodel)->getX()->size())
            {
                mass = new DiagonalMass<Vec3Types,SReal>;
                for (unsigned int i=0 ; i<massList.size() ; i++)
                {
                    ((DiagonalMass<Vec3Types,SReal>*)mass)->addMass( massList[i] );
                }
            }
            else 	//else we don't build mass...
                cerr<<"WARNING building "<<name<<" object : mass property not properly defined."<<endl;
        }
    }
    if (mass)
        parentNode->addObject(mass);
}


void PMLFemForceField::createVisualModel(StructuralComponent* body)
{
    // ADD EXTERN FACETS TO TOPOLOGY
    MultiComponent * mc = PMLTransform::generateExternalSurface(body);
    StructuralComponent  * extFacets = (StructuralComponent*) mc->getSubComponent(0);

    if (!topology)
        topology = new MeshTopology();

    Cell * pCell;
    Atom * pAtom;
    BaseMeshTopology::Quad * quad;
    BaseMeshTopology::Triangle * triangle;

    for (unsigned int i=0 ; i< extFacets->getNumberOfStructures() ; i++)
    {
        pCell = extFacets->getCell(i);
        switch(pCell->getProperties()->getType())
        {
        case StructureProperties::QUAD :
            quad = new BaseMeshTopology::Quad;
            for (unsigned int j(0) ; j<4 ; j++)
            {
                pAtom = (Atom*)(pCell->getStructure(j));
                (*quad)[j] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology)->getQuads()).push_back(*quad);
            break;

        case StructureProperties::TRIANGLE :
            triangle = new BaseMeshTopology::Triangle;
            for (unsigned int j(0) ; j<3 ; j++)
            {
                pAtom = (Atom*)(pCell->getStructure(j));
                (*triangle)[j] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((BaseMeshTopology::SeqTriangles&)((BaseMeshTopology*)topology)->getTriangles()).push_back(*triangle);
            break;

        default : break;
        }
    }

    //CREATE THE VISUAL MODEL
    OglModel * vmodel = new OglModel;

    double * color = body->getColor();
    vmodel->setColor((float)color[0], (float)color[1], (float)color[2], (float)color[3]);
    vmodel->load("","","");
    BaseMapping * mapping = new IdentityMapping< Mapping< State<Vec3Types>, MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >((MechanicalState<Vec3Types>*)mmodel, vmodel);
    parentNode->addObject(mapping);
    parentNode->addObject(vmodel);

}


//create a TetrahedronFEMForceField
void PMLFemForceField::createForceField()
{
    forcefield = new TetrahedronFEMForceField<Vec3Types>;
    //if(poisson==0.0)poisson=0.49;
    //if(young==0.0) young=5000.0;

    ((TetrahedronFEMForceField<Vec3Types>*)forcefield)->setPoissonRatio(poisson);

    ((TetrahedronFEMForceField<Vec3Types>*)forcefield)->setYoungModulus(young);

    if(deformationType== "SMALL")
        ((TetrahedronFEMForceField<Vec3Types>*)forcefield)->setMethod(0);
    if(deformationType== "LARGE")
        ((TetrahedronFEMForceField<Vec3Types>*)forcefield)->setMethod(1);
    if(deformationType== "POLAR")
        ((TetrahedronFEMForceField<Vec3Types>*)forcefield)->setMethod(2);

    parentNode->addObject(forcefield);
}


void PMLFemForceField::createCollisionModel()
{
    if (collisionsON)
    {
        tmodel = new TriangleModel;
        //lmodel = new LineModel;
        //pmodel = new PointModel;

        parentNode->addObject( tmodel);
        //parentNode->addObject( lmodel );
        //parentNode->addObject( pmodel );

        tmodel->init();
        //lmodel->init();
        //pmodel->init();
    }
}


bool PMLFemForceField::FusionBody(PMLBody* body)
{
    PMLFemForceField * femBody = (PMLFemForceField * )body;
    std::map<unsigned int, unsigned int> oldToNewIndex;

    //-----  Fusion Mechanical Model
    map<unsigned int, unsigned int>::iterator it = femBody->AtomsToDOFsIndexes.begin();
    map<unsigned int, unsigned int>::iterator itt;
    unsigned int X1size = ((MechanicalState<Vec3Types>*)mmodel)->getX()->size();
    while (it !=  femBody->AtomsToDOFsIndexes.end())
    {
        //if femBody's index doesn't exist in current list, we insert it
        if ( (itt = this->AtomsToDOFsIndexes.find( (*it).first)) == this->AtomsToDOFsIndexes.end() )
        {
            int cpt = ((MechanicalState<Vec3Types>*)mmodel)->getX()->size();
            mmodel->resize( cpt+1);
            this->AtomsToDOFsIndexes.insert(std::pair<unsigned int, unsigned int>((*it).first, cpt ));
            oldToNewIndex.insert(std::pair<unsigned int, unsigned int>((*it).second, cpt ));
            (*((MechanicalState<Vec3Types>*)mmodel)->getX())[cpt] = (*((MechanicalState<Vec3Types>*)(femBody->getMechanicalState()))->getX())[(*it).second];
        }
        else
            oldToNewIndex.insert(std::pair<unsigned int, unsigned int>((*it).second, (*itt).second) );

        it++;
    }

    //------   Fusion Topology
    BaseMeshTopology * femTopo = (BaseMeshTopology * ) (femBody->getTopology());

    //fusion tetras
    for (int i=0 ; i < femTopo->getNbTetrahedra() ; i++)
    {
        BaseMeshTopology::Tetra tet = femTopo->getTetrahedron(i);
        for (unsigned int j(0) ; j<4 ; j++)
        {
            tet[j] = oldToNewIndex[tet[j] ];
        }
        ((BaseMeshTopology::SeqTetrahedra&)((BaseMeshTopology*)topology)->getTetrahedra()).push_back(tet);
    }
    //fusion triangles
    for (int i=0 ; i < femTopo->getNbTriangles() ; i++)
    {
        BaseMeshTopology::Triangle tri = femTopo->getTriangle(i);
        for (unsigned int j(0) ; j<3 ; j++)
        {
            tri[j] = oldToNewIndex[tri[j] ];
        }
        ((BaseMeshTopology::SeqTriangles&)((BaseMeshTopology*)topology)->getTriangles()).push_back(tri);
    }
    //fusion quads
    for (int i=0 ; i < femTopo->getNbQuads() ; i++)
    {
        BaseMeshTopology::Quad qua = femTopo->getQuad(i);
        for (unsigned int j(0) ; j<4 ; j++)
        {
            qua[j] = oldToNewIndex[qua[j] ];
        }
        ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology)->getQuads()).push_back(qua);
    }


    //-------  Fusion Mass
    parentNode->removeObject(mass);
    if (mass) delete mass;
    mass = new DiagonalMass<Vec3Types,SReal>;
    parentNode->addObject(mass);
    SReal m1,m2;

    for (unsigned int i=0 ; i< ((MechanicalState<Vec3Types>*)mmodel)->getX()->size(); i++)
    {
        m1 = m2 = 0.0;
        if (massList.size() >0)
        {
            if (massList.size() == 1 && i < X1size)
                m1 = massList[0];
            else if (i < massList.size())
                m1 = massList[i];
        }
        if (femBody->massList.size() >0)
        {
            if (femBody->massList.size() == 1 )
            {
                for (unsigned int j=0 ; j<oldToNewIndex.size() ; j++)
                    if (oldToNewIndex[j] == i)
                        m2 = femBody->massList[0];
            }
            else
            {
                for (unsigned int j=0 ; j<oldToNewIndex.size() ; j++)
                    if (oldToNewIndex[j] == i)
                        m2 = femBody->massList[j];
            }
        }

        ((DiagonalMass<Vec3Types,SReal>*)mass)->addMass( m1+m2 );
        cout<<"masse noeud "<<i<<" : "<<m1+m2<<endl;
    }


    //------  Fusion Collision Model
    if (!collisionsON && femBody->collisionsON)
    {
        tmodel = femBody->getTriangleModel();
        //lmodel = femBody->getLineModel();
        //pmodel = femBody->getPointModel();
    }

    return true;
}

}
}
}
