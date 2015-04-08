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

#include "PMLStiffSpringForceField.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseMechanics/DiagonalMass.h>
#include <SofaBaseMechanics/IdentityMapping.h>


#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <PhysicalProperties/CellProperties.h>
#include <PMLTransform.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::core::objectmodel;
using namespace sofa::component::mass;
using namespace sofa::component::mapping;
using namespace sofa::component;

PMLStiffSpringForceField::PMLStiffSpringForceField(StructuralComponent* body, GNode * parent)
{
    parentNode = parent;
    //get the parameters
    collisionsON = body->getProperties()->getBool("collision");
    name = body->getProperties()->getName();

    if(body->getProperties()->getString("mass") != "")
        initMass(body->getProperties()->getString("mass"));

    if(body->getProperties()->getString("density") != "")
        initDensity(body->getProperties()->getString("density"));

    ks = body->getProperties()->getDouble("stiffness");
    kd = body->getProperties()->getDouble("damping");
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

PMLStiffSpringForceField::~PMLStiffSpringForceField()
{
}

//read the mass parameter
void PMLStiffSpringForceField::initMass(string m)
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

void PMLStiffSpringForceField::initDensity(string m)
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


// extract hexahedron edges to a list of lines
// used by createTopology method
BaseMeshTopology::Line * PMLStiffSpringForceField::hexaToLines(Cell* pCell)
{
    BaseMeshTopology::Line *lines = new BaseMeshTopology::Line[16];
    Atom *pAtom;
    int index[8];

    for (int i(0) ; i<8 ; i++)
    {
        pAtom = (Atom*)(pCell->getStructure(i));
        index[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
    }

    lines[0][0]=index[0]; lines[0][1]=index[1];
    lines[1][0]=index[1]; lines[1][1]=index[2];
    lines[2][0]=index[2]; lines[2][1]=index[3];
    lines[3][0]=index[3]; lines[3][1]=index[0];
    lines[4][0]=index[4]; lines[4][1]=index[5];
    lines[5][0]=index[5]; lines[5][1]=index[6];
    lines[6][0]=index[6]; lines[6][1]=index[7];
    lines[7][0]=index[7]; lines[7][1]=index[4];
    lines[8][0]=index[0]; lines[8][1]=index[4];
    lines[9][0]=index[1]; lines[9][1]=index[5];
    lines[10][0]=index[2]; lines[10][1]=index[6];
    lines[11][0]=index[3]; lines[11][1]=index[7];
    lines[12][0]=index[0]; lines[12][1]=index[6];
    lines[13][0]=index[1]; lines[13][1]=index[7];
    lines[14][0]=index[2]; lines[14][1]=index[4];
    lines[15][0]=index[3]; lines[15][1]=index[5];

    return lines;
}

// extract tetrahedron edges to a list of lines
// used by createTopology method
BaseMeshTopology::Line * PMLStiffSpringForceField::tetraToLines(Cell* pCell)
{
    BaseMeshTopology::Line *lines = new BaseMeshTopology::Line[6];
    Atom *pAtom;
    int index[8];

    for (int i(0) ; i<4 ; i++)
    {
        pAtom = (Atom*)(pCell->getStructure(i));
        index[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
    }
    unsigned int cpt=0;
    for (unsigned int l1=0 ; l1<4 ; l1++)
    {
        for (unsigned int l2=l1+1 ; l2<4 ; l2++)
        {
            lines[cpt][0] = index[l1];
            lines[cpt][1] = index[l2];
            cpt++;
        }
    }

    return lines;
}

// extract triangle edges to a list of lines
// used by createTopology method
BaseMeshTopology::Line * PMLStiffSpringForceField::triangleToLines(Cell* pCell)
{
    BaseMeshTopology::Line *lines = new BaseMeshTopology::Line[3];
    Atom *pAtom;
    int index[3];

    for (int i(0) ; i<3 ; i++)
    {
        pAtom = (Atom*)(pCell->getStructure(i));
        index[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
    }
    unsigned int cpt=0;
    for (unsigned int l1=0 ; l1<4 ; l1++)
    {
        for (unsigned int l2=l1+1 ; l2<4 ; l2++)
        {
            lines[cpt][0] = index[l1];
            lines[cpt][1] = index[l2];
            cpt++;
        }
    }
    return lines;
}

// extract quad edges to a list of lines
// used by createTopology method
BaseMeshTopology::Line * PMLStiffSpringForceField::quadToLines(Cell* pCell)
{
    BaseMeshTopology::Line *lines = new BaseMeshTopology::Line[4];
    Atom *pAtom;
    int index[4];

    for (int i(0) ; i<4 ; i++)
    {
        pAtom = (Atom*)(pCell->getStructure(i));
        index[i] = AtomsToDOFsIndexes[pAtom->getIndex()];
    }
    for (unsigned int l1=0 ; l1<4 ; l1++)
    {
        lines[l1][0] = index[l1];
        lines[l1][1] = index[(l1+1)%4];
    }
    return lines;
}




Vector3 PMLStiffSpringForceField::getDOF(unsigned int index)
{
    return ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue()[index];
}

//creation of the mechanical model
//each pml atom constituing the body correspond to a DOF
void PMLStiffSpringForceField::createMechanicalState(StructuralComponent* body)
{
    mmodel = New<MechanicalObject<Vec3Types> >();
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    SReal pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        ((MechanicalState<Vec3Types>*)mmodel.get())->writePositions()[i] = Vector3(pos[0],pos[1],pos[2]);
    }

    parentNode->addObject(mmodel);

}


// creation of the topolgy
// topology constituted exclusively by tetrahedrons
// --> if there is hexahedrons, they are tesselated in 5 tetrahedron
void PMLStiffSpringForceField::createTopology(StructuralComponent* body)
{
    topology = New<MeshTopology>();
    ((BaseMeshTopology*)topology.get())->clear();

    unsigned int p, nbCells = body->getNumberOfCells();
    BaseMeshTopology::Line * lines;
    BaseMeshTopology::Quad * quad;
    Cell * pCell;

    //for each pml cell, build 1 or 5 tetrahedrons
    for (unsigned int cid(0) ; cid<nbCells ; cid++)
    {
        pCell = body->getCell(cid);
        switch(pCell->getProperties()->getType())
        {

        case StructureProperties::HEXAHEDRON :
            lines = hexaToLines(pCell);
            for (p=0 ; p<16 ; p++)
                ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(lines[p]);
            break;
        case StructureProperties::TETRAHEDRON :
            lines = hexaToLines(pCell);
            for (p=0 ; p<6 ; p++)
                ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(lines[p]);
            break;
        case StructureProperties::TRIANGLE :
            lines = triangleToLines(pCell);
            for (p=0 ; p<3 ; p++)
                ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(lines[p]);
            break;
        case StructureProperties::QUAD :
            lines = quadToLines(pCell);
            for (p=0 ; p<4 ; p++)
                ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(lines[p]);
            quad = new BaseMeshTopology::Quad;
            for (p=0 ; p<4 ; p++)
                (*quad)[p] = AtomsToDOFsIndexes[pCell->getStructure(p)->getIndex()];
            ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology.get())->getQuads()).push_back(*quad);
            break;
        case StructureProperties::LINE :
            lines = new BaseMeshTopology::Line;
            (*lines)[0] = AtomsToDOFsIndexes[pCell->getStructure(0)->getIndex()];
            (*lines)[1] = AtomsToDOFsIndexes[pCell->getStructure(1)->getIndex()];
            ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(*lines);
            break;

        default : break;
        }
    }

    //ELIMINATE DOUBLONS
    std::vector<BaseMeshTopology::Line>::iterator it1 = ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).begin();
    std::vector<BaseMeshTopology::Line>::iterator it2, tmp;

    while(it1 != ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).end() )
    {
        it2=it1;
        it2++;
        while(it2 != ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).end() )
        {
            if ( ((*it1)[0] == (*it2)[0] && (*it1)[1] == (*it2)[1]) || ((*it1)[0] == (*it2)[1] && (*it1)[1] == (*it2)[0]) )
            {
                tmp = it2;
                tmp--;
                ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).erase(it2);
                it2=tmp;
            }

            it2++;
        }
        it1++;
    }

    parentNode->addObject(topology);
}


//creation of the mass
//normaly there 1 value OR nbDOFs values (OR 0 if not specified)
void PMLStiffSpringForceField::createMass(StructuralComponent* body)
{
    //if no mass specified...
    if (massList.size() == 0)
    {
        //...normally density is!
        if (density.size() != 0)
        {
            //BUILDING WITH DENSITY PROPERTY
            if (density.size() > 1 && density.size() != ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size())
            {
                cerr<<"WARNING building "<<name<<" object : density property not properly defined."<<endl;
                return;
            }
            else
            {
                //init the mass list
                for (unsigned int i=0 ; i<((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size() ; i++)
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
                mass = New<DiagonalMass<Vec3Types,SReal> >();
                for (unsigned int im=0 ; im<massList.size() ; im++)
                {
                    ((DiagonalMass<Vec3Types,SReal>*)mass.get())->addMass( massList[im] );
                }
            }
        }
    } //BUILDING WITH MASS PROPERTY
    else
    {
        //if there is 1 value --> uniform mass for all the model
        if (massList.size() == 1)
        {
            mass = New<UniformMass<Vec3Types,SReal> >();
            ((UniformMass<Vec3Types,SReal>*)mass.get())->setMass( massList[0] );
        }
        else
        {
            //if there nbDofs values --> diagonal mass (one value for each dof)
            if (massList.size() == ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size())
            {
                mass = New<DiagonalMass<Vec3Types,SReal> >();
                for (unsigned int i=0 ; i<massList.size() ; i++)
                {
                    ((DiagonalMass<Vec3Types,SReal>*)mass.get())->addMass( massList[i] );
                }
            }
            else 	//else we don't build mass...
                cerr<<"WARNING building "<<name<<" object : mass property not properly defined."<<endl;
        }
    }
    if (mass)
        parentNode->addObject(mass);
}


void PMLStiffSpringForceField::createVisualModel(StructuralComponent* body)
{
    // ADD EXTERN FACETS TO TOPOLOGY
    MultiComponent * mc = PMLTransform::generateExternalSurface(body);
    StructuralComponent  * extFacets = (StructuralComponent*) mc->getSubComponent(0);

    if (!topology)
        topology = New<MeshTopology>();

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
            ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology.get())->getQuads()).push_back(*quad);
            break;

        case StructureProperties::TRIANGLE :
            triangle = new BaseMeshTopology::Triangle;
            for (unsigned int j(0) ; j<3 ; j++)
            {
                pAtom = (Atom*)(pCell->getStructure(j));
                (*triangle)[j] = AtomsToDOFsIndexes[pAtom->getIndex()];
            }
            ((BaseMeshTopology::SeqTriangles&)((BaseMeshTopology*)topology.get())->getTriangles()).push_back(*triangle);
            break;

        default : break;
        }
    }

    //CREATE THE VISUAL MODEL
    OglModel::SPtr vmodel = New<OglModel>();

    double * color = body->getColor();
    vmodel->setColor((float)color[0], (float)color[1], (float)color[2], (float)color[3]);
    vmodel->load("","","");
    BaseMapping::SPtr mapping = New<IdentityMapping< Vec3Types, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > >();
    ((Mapping< Vec3Types, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >*)mapping.get())->setModels(((MechanicalState<Vec3Types>*)mmodel.get()), vmodel.get());
    parentNode->addObject(mapping);
    parentNode->addObject(vmodel);

}


//create a TetrahedronFEMForceField
void PMLStiffSpringForceField::createForceField()
{
    Sforcefield = New<MeshSpringForceField<Vec3Types> >();
    if (kd==0.0)kd=5.0;
    if (ks==0.0)ks=500.0;
    Sforcefield->setLinesDamping(kd);
    Sforcefield->setLinesStiffness(ks);
    parentNode->addObject(dynamic_cast<BaseObject*>(Sforcefield.get()));
}


void PMLStiffSpringForceField::createCollisionModel()
{
    if (collisionsON)
    {
        tmodel = New<TriangleModel>();
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


bool PMLStiffSpringForceField::FusionBody(PMLBody* body)
{
    PMLStiffSpringForceField * femBody = (PMLStiffSpringForceField * )body;
    std::map<unsigned int, unsigned int> oldToNewIndex;

    //-----  Fusion Mechanical Model
    map<unsigned int, unsigned int>::iterator it = femBody->AtomsToDOFsIndexes.begin();
    map<unsigned int, unsigned int>::iterator itt;
    unsigned int X1size = ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size();
    while (it !=  femBody->AtomsToDOFsIndexes.end())
    {
        //if femBody's index doesn't exist in current list, we insert it
        if ( (itt = this->AtomsToDOFsIndexes.find( (*it).first)) == this->AtomsToDOFsIndexes.end() )
        {
            int cpt = ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size();
            mmodel->resize( cpt+1);
            this->AtomsToDOFsIndexes.insert(std::pair<unsigned int, unsigned int>((*it).first, cpt ));
            oldToNewIndex.insert(std::pair<unsigned int, unsigned int>((*it).second, cpt ));
            ((MechanicalState<Vec3Types>*)mmodel.get())->writePositions()[cpt] = ((MechanicalState<Vec3Types>*)(femBody->getMechanicalState().get()))->read(core::ConstVecCoordId::position())->getValue()[(*it).second];
        }
        else
            oldToNewIndex.insert(std::pair<unsigned int, unsigned int>((*it).second, (*itt).second) );

        it++;
    }

    //------   Fusion Topology
    BaseMeshTopology * femTopo = (BaseMeshTopology * ) (femBody->getTopology().get());

    //fusion lines
    for (int i=0 ; i < femTopo->getNbLines() ; i++)
    {
        BaseMeshTopology::Line line = femTopo->getLine(i);
        for (unsigned int j(0) ; j<2 ; j++)
        {
            line[j] = oldToNewIndex[line[j] ];
        }
        ((BaseMeshTopology::SeqLines&)((BaseMeshTopology*)topology.get())->getLines()).push_back(line);
    }
    //fusion triangles
    for (int i=0 ; i < femTopo->getNbTriangles() ; i++)
    {
        BaseMeshTopology::Triangle tri = femTopo->getTriangle(i);
        for (unsigned int j(0) ; j<3 ; j++)
        {
            tri[j] = oldToNewIndex[tri[j] ];
        }
        ((BaseMeshTopology::SeqTriangles&)((BaseMeshTopology*)topology.get())->getTriangles()).push_back(tri);
    }
    //fusion quads
    for (int i=0 ; i < femTopo->getNbQuads() ; i++)
    {
        BaseMeshTopology::Quad qua = femTopo->getQuad(i);
        for (unsigned int j(0) ; j<4 ; j++)
        {
            qua[j] = oldToNewIndex[qua[j] ];
        }
        ((BaseMeshTopology::SeqQuads&)((BaseMeshTopology*)topology.get())->getQuads()).push_back(qua);
    }


    //-------  Fusion Mass
    parentNode->removeObject(mass);
    mass = New<DiagonalMass<Vec3Types,SReal> >();
    parentNode->addObject(mass);
    SReal m1,m2;

    for (unsigned int i=0 ; i< ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue().size(); i++)
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

        ((DiagonalMass<Vec3Types,SReal>*)mass.get())->addMass( m1+m2 );
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
