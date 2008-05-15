/***************************************************************************
                              PhysicalModel.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2007/03/26 07:20:54 $
    Version           : $Revision: 1.33 $
 ***************************************************************************/
 
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// Reading XML
#include <libxml/parser.h>
#include <libxml/tree.h>

// Other includes
#include "PhysicalModel.h"
//#include "Object3D.h"
#include "Atom.h"
#include "Cell.h"
#include "CellProperties.h"
#include "MultiComponent.h"
#include "StructuralComponent.h"

#include "AbortException.h"

// --------------- static member initialization -----------------
// Version #
const std::string PhysicalModel::VERSION = "0.6 - 20 july 2006";


//--------------- Constructor/Destructor ------------------------------
PhysicalModel::PhysicalModel() {
    init();
}

PhysicalModel::PhysicalModel(const char * fileName, PtrToSetProgressFunction pspf){
    init();
    setProgressFunction = pspf;

    // load from the xml file
    xmlRead(fileName);
}

// --------------- destructor ---------------
PhysicalModel::~PhysicalModel() {
    clear();
}

// --------------- init ---------------
void PhysicalModel::init() {
    name = "";
    exclusiveComponents = NULL;
    informativeComponents = NULL;
    atoms = NULL;
    setProgressFunction = NULL;
    cellIndexOptimized = true; //always hopeful!
}

// --------------- setProgress ---------------
void PhysicalModel::setProgress(const float donePercentage) {
    if (setProgressFunction != NULL)
        setProgressFunction(donePercentage);
}

// --------------- clear ---------------
void PhysicalModel::clear() {
    name = "";
    if (atoms)
        delete atoms;
    if (informativeComponents)
        delete informativeComponents;
	if (exclusiveComponents)
        delete exclusiveComponents;
    atoms = NULL;
    exclusiveComponents = NULL;
    informativeComponents = NULL;
    // reset all the unique indexes
    AtomProperties::resetUniqueIndex();
    CellProperties::resetUniqueIndex();
    atomMap.clear();
    cellMap.clear();
}

// --------------- getNumberOfCells   ---------------
unsigned int PhysicalModel::getNumberOfCells() const {
    unsigned int nrOfCells = 0;
    if (exclusiveComponents)
        nrOfCells += exclusiveComponents->getNumberOfCells();
    if (informativeComponents)
        nrOfCells += informativeComponents->getNumberOfCells();

    //@@@ Mahnu, pourquoi il faut ajouter les informatives dans nrOfCells?
    //@@@ est-ce qu'on ne compte pas les cellules plusieurs fois??
    // Because so the nr of cells could be used to generate a nice progress bar...
    // What for? I do have a great computer now!!! Speed == very high!!
    // think scalability ... down!
    // What?
    // if you ever come down to this comment, Matt, I meant "think about people who don't have the last technology (like everyone after a buy of more than one day!)
    return nrOfCells;
}


// --------------- optimizeIndexes ---------------
void PhysicalModel::optimizeIndexes(MultiComponent * mc, unsigned int * index) {
    Component *c;
    StructuralComponent *sc;
    Cell *cell;

    for (unsigned int i=0; i<mc->getNumberOfSubComponents(); i++) {
        c = mc->getSubComponent(i);
        if (c->isInstanceOf("MultiComponent"))
            optimizeIndexes((MultiComponent *)c,index);
        else {
            if (c->isInstanceOf("StructuralComponent")) {
                sc = (StructuralComponent *) c;
                // check all cells
                for (unsigned int j=0; j<sc->getNumberOfStructures(); j++) {
                    if (sc->getStructure(j)->isInstanceOf("Cell")) {
                        cell = (Cell *) sc->getStructure(j);
                        // if this is the sc that make the cell print its data, change cell index
                        if (cell->makePrintData(sc)) {
                            cell->setIndex(*index);
                            *index = (*index) + 1;
                        }
                    }
                }
            }
        }
    }
}

void PhysicalModel::optimizeIndexes() {
    // to optimize the indexes: do as if it was a print/read operation (same order
    // and change the cell index (as everyone is linked with ptrs, that should not
    // change anything else

    // first: the atoms
    if (atoms) {
        for (unsigned int i=0; i<atoms->getNumberOfStructures(); i++)
            ((Atom *) atoms->getStructure(i))->setIndex(i);
    }

    // then the cells
    unsigned int newIndex =  0;
    if (exclusiveComponents) {
        optimizeIndexes(exclusiveComponents, &newIndex);
    }
    if (informativeComponents) {
        optimizeIndexes(informativeComponents, &newIndex);
    }

}

// --------------- xmlPrint   ---------------
void PhysicalModel::xmlPrint(std::ostream &o, bool opt) {

    // should we optimize the cell indexes ?
    if (!cellIndexOptimized && opt) {
        optimizeIndexes();
    }

    // print out the whole thing
    o << "<!-- physical model is a generic representation for 3D physical model (FEM, spring mass network, phymulob...) --> " <<std::endl;
    o << "<physicalModel";
    if (getName() != "")
        o<< " name=\"" << getName().c_str() << "\"";
    if (atoms)
        o << " nrOfAtoms=\"" << atoms->getNumberOfStructures() << "\"" << std::endl;
    if (exclusiveComponents)
        o << " nrOfExclusiveComponents=\"" << exclusiveComponents->getNumberOfSubComponents() << "\"" << std::endl;
    if (informativeComponents)
        o << " nrOfInformativeComponents=\"" << informativeComponents->getNumberOfSubComponents() << "\"" << std::endl;
    o << " nrOfCells=\"" << getNumberOfCells() << "\"" << std::endl;
    o << ">" << std::endl;
    o << "<!-- list of atoms: -->" << std::endl;
    o << "<atoms>" << std::endl;
    if (atoms)
        atoms->xmlPrint(o);
    o << "</atoms>" << std::endl;
    o << "<!-- list of exclusive components : -->" << std::endl;
    o << "<exclusiveComponents>" << std::endl;
    if (exclusiveComponents)
        exclusiveComponents->xmlPrint(o);
    o << "</exclusiveComponents>" << std::endl;
    o << "<!-- list of informative components : -->" << std::endl;
    o << "<informativeComponents>" << std::endl;
    if (informativeComponents)
        informativeComponents->xmlPrint(o);
    o << "</informativeComponents>" << std::endl;
    o << "</physicalModel>" << std::endl;

}

// --------------- xmlRead   ---------------
void PhysicalModel::xmlRead(const char * n){
    // clear all the current data
    clear();

    //static const char* xmlFile = 0;
    static bool isInit = false;

    //if (!isInit) {

		// this initialize the library and check potential ABI mismatches
		// between the version it was compiled for and the actual shared
		// library used.
		LIBXML_TEST_VERSION

		// the resulting document tree 
		xmlDocPtr doc; 
		//the pointer to the root node of the document
		xmlNodePtr root;

		doc = xmlParseFile(n);
		if (doc == NULL) {
			std::cerr << "Failed to open " << n << std::endl;
			return ;
		}
		
		root = xmlDocGetRootElement(doc);
		if (root == NULL) {
			std::cerr << "empty document" << std::endl;
			xmlFreeDoc(doc);
			return ;
		}

		//build the physicalModel, parsing the xml tree
		if (!parseTree(root)){
			std::cerr << "failed to read the xml tree" << std::endl;
			xmlFreeDoc(doc);
			return ;
		}

		//free the xml
		xmlFreeDoc(doc);
		xmlCleanupParser();
		xmlMemoryDump();

    //}

    isInit = true;
}


// ------------------ parse tree ------------------
bool PhysicalModel::parseTree(xmlNodePtr root)
{
	if (xmlStrcmp(root->name,(const xmlChar*)"physicalModel")){
		std::cerr << "failed to read the physicalModel" << std::endl;
		return false;
	}

	//read the physicalModel properties
	xmlChar *pname = xmlGetProp(root, (const xmlChar*) "name");
	if(pname) setName((char*)pname);

	/*xmlChar *pnrCells = xmlGetProp(root, (const xmlChar*) "nrOfCells");
	if (pnrCells){
		optimizedCellList.reserve( atoi( (char*)pnrCells ) );
		cellIndexOptimized = true;
	} else
		cellIndexOptimized = false;
	*/

	//get the pointer on atoms
	xmlNodePtr atomsPtr = root->xmlChildrenNode;
	while( atomsPtr && xmlStrcmp(atomsPtr->name,(const xmlChar*)"atoms")) atomsPtr = atomsPtr->next ;
	if (!atomsPtr || xmlStrcmp(atomsPtr->name,(const xmlChar*)"atoms"))
	{
		std::cerr << "failed to read the atoms" << std::endl;
		return false;
	} 
	else 
	{
		//if we found the pointer on atoms, parse the sub tree and build the atoms
		parseAtoms(atomsPtr);
	}

	//get the pointer on exclusiveComponents
	xmlNodePtr exCompPtr = atomsPtr->next;
	while( exCompPtr && xmlStrcmp(exCompPtr->name,(const xmlChar*)"exclusiveComponents")) exCompPtr = exCompPtr->next ;

	if (!exCompPtr || xmlStrcmp(exCompPtr->name,(const xmlChar*)"exclusiveComponents"))
	{
		std::cerr << "failed to read the exclusiveComponents" << std::endl;
		return false;
	} 
	else 
	{
		//if we found the pointer on exclusiveComponents, parse the sub tree and build the exclusiveComponents
		//exclusiveComponents = new MultiComponent(this);
		xmlNodePtr mcNode = exCompPtr->children;
		//get the pointer on the multicomponent child
		while( mcNode && xmlStrcmp(mcNode->name,(const xmlChar*)"multiComponent")) mcNode = mcNode->next ;
		if (!mcNode){
			std::cerr<<"error : no exclusive components found."<<std::endl;
			return false;
		}
		MultiComponent * mc = new MultiComponent(this);
		xmlChar *pname = xmlGetProp(mcNode, (const xmlChar*) "name");
		if(pname) mc->setName(std::string((char*)pname));

		parseComponents(mcNode, mc, true);

		setExclusiveComponents(mc);
	}

	//get the pointer on informativeComponents
	xmlNodePtr infCompPtr = exCompPtr->next;
	while( infCompPtr && xmlStrcmp(infCompPtr->name,(const xmlChar*)"informativeComponents")) infCompPtr = infCompPtr->next ;
	if (!infCompPtr || xmlStrcmp(infCompPtr->name,(const xmlChar*)"informativeComponents"))
	{
		std::cerr << "failed to read the informativeComponents" << std::endl;
		return false;
	} 
	else 
	{
		//if we found the pointer on informativeComponents, parse the sub tree and build the informativeComponents
		informativeComponents = new MultiComponent(this);
		xmlNodePtr mcNode = infCompPtr->children;
		//get the pointer on the multicomponent child
		while( mcNode && xmlStrcmp(mcNode->name,(const xmlChar*)"multiComponent")) mcNode = mcNode->next ;
		if (!mcNode){
			//if no child found, there is no informative components
			delete informativeComponents;
			informativeComponents = NULL;
			return true;
		}
		xmlChar *pname = xmlGetProp(mcNode, (const xmlChar*) "name");
		if(pname) informativeComponents->setName(std::string((char*)pname));

		parseComponents(mcNode, informativeComponents, false);
	}

	return true;
}


// ------------------ parse atoms ------------------
bool PhysicalModel::parseAtoms(xmlNodePtr atomsRoot)
{
	//parse the content of atoms
	for (xmlNodePtr child = atomsRoot->xmlChildrenNode; child != NULL; child = child->next)
	{
		//get the pointer on structuralComponent
		if (!xmlStrcmp(child->name,(const xmlChar*)"structuralComponent")) 
		{
			StructuralComponent * sc = new StructuralComponent(this, child);
			//parse the content of structuralComponent
			for (xmlNodePtr SCchild = child->xmlChildrenNode; SCchild != NULL; SCchild = SCchild->next)
			{
				//get the pointer on nrOfStructures
				if (!xmlStrcmp(SCchild->name,(const xmlChar*)"nrOfValues")) {
					xmlChar * prop = xmlGetProp(SCchild, (xmlChar*)("value"));
					unsigned int val = atoi((char*)prop);
					sc->plannedNumberOfStructures(val);
				}
				//get the pointers on atom
				if (!xmlStrcmp(SCchild->name,(const xmlChar*)"atom")) {
					Atom * newatom = new Atom(this, SCchild);
					sc->addStructure(newatom);
				}
			}
			this->setAtoms(sc);
		}
		if (!xmlStrcmp(child->name,(const xmlChar*)"atom")) 
		{
			Atom * newatom = new Atom(this, child);
			this->addAtom(newatom);
		}
	}

	return true;
}


// ------------------ parse Components ------------------
bool PhysicalModel::parseComponents(xmlNodePtr root, Component * father, bool isExclusive)
{
	for (xmlNodePtr child = root->xmlChildrenNode; child != NULL; child = child->next)
	{
		//read and build a MultiComponent
		if (!xmlStrcmp(child->name,(const xmlChar*)"multiComponent")) 
		{
			MultiComponent * mc = new MultiComponent(this);
			mc->setExclusive(isExclusive);

			xmlChar *pname = xmlGetProp(child, (const xmlChar*) "name");
			if(pname) mc->setName(std::string((char*)pname));

			((MultiComponent*)father)->addSubComponent(mc);

			//parse the multicomponent subtree
			parseComponents(child, mc, isExclusive);
		}

		//read and build a structuralComponent
		if (!xmlStrcmp(child->name,(const xmlChar*)"structuralComponent")) 
		{
			StructuralComponent * sc = new StructuralComponent(this, child);
			sc->setExclusive(isExclusive);

			xmlNodePtr SCchild = child->xmlChildrenNode;
			while( SCchild != NULL && xmlStrcmp(SCchild->name,(const xmlChar*)"nrOfValues"))
				SCchild = SCchild->next;
			if (SCchild) {
				xmlChar * prop = xmlGetProp(SCchild, (xmlChar*)("value"));
				unsigned int val = atoi((char*)prop);
				sc->plannedNumberOfStructures(val);
			}
			((MultiComponent*)father)->addSubComponent(sc);

			//parse the structuralcomponent subtree
			parseComponents(child, sc, isExclusive);
		}

		//read and build a cell
		if (!xmlStrcmp(child->name,(const xmlChar*)"cell")) 
		{
			StructureProperties::GeometricType gtype = StructureProperties::INVALID;
			//find properties node
			xmlNodePtr cchild = child->xmlChildrenNode;
			while ( cchild != NULL && xmlStrcmp(cchild->name,(const xmlChar*)"cellProperties"))
				cchild = cchild->next;
			if (cchild)
			{
				//search the type attribute
				xmlChar *ptype = xmlGetProp(cchild, (const xmlChar*) "type");
				if (ptype){
					if (!xmlStrcmp(ptype,(const xmlChar*)"TRIANGLE"))
						gtype = StructureProperties::TRIANGLE; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"QUAD"))
						gtype = StructureProperties::QUAD; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"TETRAHEDRON"))
						gtype = StructureProperties::TETRAHEDRON; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"HEXAHEDRON"))
						gtype = StructureProperties::HEXAHEDRON; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"LINE"))
						gtype = StructureProperties::LINE; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"WEDGE"))
						gtype = StructureProperties::WEDGE; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"POLY_LINE"))
						gtype = StructureProperties::POLY_LINE; 
					if (!xmlStrcmp(ptype,(const xmlChar*)"POLY_VERTEX"))
						gtype = StructureProperties::POLY_VERTEX; 
				}
			}
			Cell * c = new Cell(this,gtype, child);
			c->setExclusive(isExclusive);
			if (cellIndexOptimized){
				std::GlobalIndexStructurePair pair(c->getIndex(), c);
				this->addGlobalIndexCellPair(pair);
			}
			((StructuralComponent*)father)->addStructure(c, false);

			//parse the cell subtree
			parseComponents(child, c, isExclusive);
		}

		//find the atomRef nodes
		if (!xmlStrcmp(child->name,(const xmlChar*)"atomRef")) 
		{
			xmlChar * prop = xmlGetProp(child, (xmlChar*)("index"));
			unsigned int val = atoi((char*)prop);
			((StructuralComponent*)father)->addStructure(this->getAtom(val));
		}

		//find the cellRef nodes
		if (!xmlStrcmp(child->name,(const xmlChar*)"cellRef")) 
		{
			xmlChar * prop = xmlGetProp(child, (xmlChar*)("index"));
			unsigned int val = atoi((char*)prop);
			((StructuralComponent*)father)->addStructure(this->getCell(val), false);
		}
			
	}
	return true;
}


    
// ------------------ getComponentByName ------------------
Component * PhysicalModel::getComponentByName(const std::string n) {
    // look for the component in the informative and exclusive component
    Component * foundC;
    foundC = exclusiveComponents->getComponentByName(n);
    if (!foundC)
        foundC = informativeComponents->getComponentByName(n);
    return foundC;
}

// ----------------------- setAtoms ------------------
void PhysicalModel::setAtoms(StructuralComponent *sc, bool deleteOld) {
    Atom *a;

    if (sc->composedBy()==StructuralComponent::ATOMS) {
        if (atoms && deleteOld)
            delete atoms;
        atoms = sc;
        // register all the atoms in the map, and tell the atoms about its new status
        for (unsigned int i=0; i<sc->getNumberOfStructures();i++) {
            a = (Atom *) sc->getStructure(i);
	    a->getProperties()->setPhysicalModel(this);
            addGlobalIndexAtomPair(std::GlobalIndexStructurePair(a->getIndex(),a));
        }
    }
}

// ----------------------- addAtom ------------------
bool PhysicalModel::addAtom(Atom *newA) {
    // register the atom in the map if possible
    if (atoms && addGlobalIndexAtomPair(std::GlobalIndexStructurePair(newA->getIndex(), newA))) {
        // add the atom in the atom structural component
        atoms->addStructure(newA);
        return true;
    }
    else
        return false; // atom does not have a unique index
}

// ----------------------- addGlobalIndexAtomPair ------------------
bool PhysicalModel::addGlobalIndexAtomPair(std::GlobalIndexStructurePair p) {
    std::GlobalIndexStructureMapIterator mapIt;
    // check if the atom's index is unique
    mapIt = atomMap.find(p.first);
    
    // if the index was found, one can not add the atom
    if (mapIt!=atomMap.end())
        return false;
    
    // if the atom is present in the map then replace the pair <atomIndex, Atom*>
    mapIt = atomMap.begin();
    while (mapIt != atomMap.end() && mapIt->second!=p.second) {
        mapIt++;
    }

    // if found then remove the pair
    if (mapIt!=atomMap.end())
        atomMap.erase(mapIt);

    // insert or re-insert (and return true if insertion was ok)
    return atomMap.insert(p).second;
}

// ----------------------- addGlobalIndexCellPair ------------------
bool PhysicalModel::addGlobalIndexCellPair(std::GlobalIndexStructurePair p) {
    std::GlobalIndexStructureMapIterator mapIt;
    // check if the cell index is unique
    mapIt = cellMap.find(p.first);
    
    // if the index was found, one can not add the cell
    if (mapIt!=cellMap.end())
        return false;
    
    // if the cell is present in the map then replace the pair <cellIndex, Cell*>
    mapIt = cellMap.begin();
    while (mapIt != cellMap.end() && mapIt->second!=p.second) {
        mapIt++;
    }

    // if found then remove the pair
    if (mapIt!=cellMap.end())
        cellMap.erase(mapIt);

    // insert or re-insert
    bool insertionOk = cellMap.insert(p).second;
    
    // is that optimized?
    cellIndexOptimized = cellIndexOptimized && ((Cell *)p.second)->getIndex() == optimizedCellList.size();
    if (cellIndexOptimized)
        optimizedCellList.push_back((Cell *)p.second);
    
    // insert or re-insert (and return true if insertion was ok)
    return insertionOk;
}

// ----------------------- setAtomPosition ------------------
void PhysicalModel::setAtomPosition(Atom *atom, const SReal pos[3]) {
    atom->setPosition(pos);
}

// ----------------------- setExclusiveComponents ------------------
void PhysicalModel::setExclusiveComponents(MultiComponent *mc) {
    if (exclusiveComponents)
        delete exclusiveComponents;
    exclusiveComponents = mc;
}

// ----------------------- setInformativeComponents ------------------
void PhysicalModel::setInformativeComponents(MultiComponent *mc) {
    if (informativeComponents)
        delete informativeComponents;
    informativeComponents = mc;
}

// ----------------------- getNumberOfExclusiveComponents ------------------
unsigned int PhysicalModel::getNumberOfExclusiveComponents() const {
    if (!exclusiveComponents)
        return 0;
    else
        return exclusiveComponents->getNumberOfSubComponents();
}

// ----------------------- getNumberOfInformativeComponents ------------------
unsigned int PhysicalModel::getNumberOfInformativeComponents() const {
    if (!informativeComponents)
        return 0;
    else
        return informativeComponents->getNumberOfSubComponents();
}

// ----------------------- getNumberOfAtoms ------------------
unsigned int PhysicalModel::getNumberOfAtoms() const {
    if (!atoms)
        return 0;
    else
        return atoms->getNumberOfStructures();
}

// ----------------------- getExclusiveComponent ------------------
Component * PhysicalModel::getExclusiveComponent(const unsigned int i) const {
    if (!exclusiveComponents)
        return 0;
    else
        return exclusiveComponents->getSubComponent(i);
}

// ----------------------- getInformativeComponent ------------------
Component * PhysicalModel::getInformativeComponent(const unsigned int i ) const {
    if (!informativeComponents)
        return 0;
    else
        return informativeComponents->getSubComponent(i);
}


// ----------------------- exportAnsysMesh ------------------
void PhysicalModel::exportAnsysMesh(std::string filename) {
    unsigned int i,j,k;

    unsigned int nbPoints = this->getNumberOfAtoms();
    unsigned int nbElements = this->getExclusiveComponent(0)->getNumberOfCells();


    //--- Ecriture des noeuds

    // fprintf are used, since this method is copied from an old C routine...
    FILE * nodeFile = NULL;
    nodeFile = fopen((filename + ".node").c_str(),"w");
    if (!nodeFile) {
        std::cerr << "Error in PhysicalModel::exportAnsysMesh : unable to create .node output file" << std::endl;
        return;
    }

    for(i=0; i<nbPoints; i++) {
        SReal pos[3];

        // WARNING : indexes are in base 1 !!!!
        j = getAtom(i)->getIndex() + 1;

        // coordinates of this node
        getAtom(i)->getPosition(pos);

        fprintf(nodeFile, "%8d %+3.8E     %+3.8E     %+3.8E    \n", j, pos[0], pos[1], pos[2]);
    }

    fclose(nodeFile);




    //--- Ecriture des elements : exlusive cells

    // fprintf are used, since this method is copied from an old C routine...
    FILE * elemFile = NULL;
    elemFile = fopen((filename + ".elem").c_str(),"w");
    if (!elemFile) {
        std::cerr << "Error in PhysicalModel::exportAnsysMesh : unable to create .elem output file" << std::endl;
        return;
    }

    int MAT, TYPE;

    for(i=0; i<nbElements; i++) {

        // get the cell
        Cell * cell = this->getExclusiveComponent(0)->getCell(i);
        j = cell->getIndex() + 1;

        switch (cell->getType()) {
        case StructureProperties::HEXAHEDRON:

            // I,J,K,L,M,N,O,P,MAT,TYPE,REAL,SECNUM,ESYS,IEL
            //
            // Format hex:
            //   I,J,K,L,M,N,O,P                = indices des noeuds
            //   MAT,TYPE,REAL,SECNUM et ESYS   = attributes numbers
            //   SECNUM                         = beam section number
            //   IEL                            = element number

            MAT  = 1;
            TYPE = 1;

            for (k=0; k<cell->getNumberOfStructures(); k++) {
                fprintf(elemFile, " %5d", cell->getStructure(k)->getIndex() + 1);
            }
            fprintf(elemFile, " %5d %5d     1     1     0 %5d\n", MAT, TYPE, j);

            break;


        case StructureProperties::WEDGE:

            // I,J,K,L,M,N,O,P,MAT,TYPE,REAL,SECNUM,ESYS,IEL
            //
            // Format prism: on repete les noeuds 3 et 7:
            //   I,J,K,K,M,N,O,O                = indices des noeuds
            //   MAT,TYPE,REAL,SECNUM et ESYS   = attributes numbers
            //   SECNUM                         = beam section number
            //   IEL                            = element number
            MAT  = 1;
            TYPE = 1;

            fprintf(elemFile, " %5d", cell->getStructure(0)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(1)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);

            fprintf(elemFile, " %5d", cell->getStructure(3)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(4)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(5)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(5)->getIndex() + 1);

            fprintf(elemFile, " %5d %5d     1     1     0 %5d\n", MAT, TYPE, j);

            break;


        case StructureProperties::TETRAHEDRON:

            MAT  = 1;
            TYPE = 2;

            fprintf(elemFile, " %5d", cell->getStructure(0)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(1)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);
            fprintf(elemFile, "     0     0     0     0");

            fprintf(elemFile, " %5d %5d     1     1     0 %5d\n", MAT, TYPE, j);
            break;



        case StructureProperties::QUAD:
            // I,J,K,L,M,N,O,P,MAT,TYPE,REAL,SECNUM,ESYS,IEL
            //
            // Format quad:
            //   I,J,K,K,L,L,L,L                = indices des noeuds
            //   MAT,TYPE,REAL,SECNUM et ESYS   = attributes numbers
            //   SECNUM                         = beam section number
            //   IEL                            = element number

            MAT  = 1;
            TYPE = 1;

            fprintf(elemFile, " %5d", cell->getStructure(0)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(1)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(2)->getIndex() + 1);

            fprintf(elemFile, " %5d", cell->getStructure(3)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(3)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(3)->getIndex() + 1);
            fprintf(elemFile, " %5d", cell->getStructure(3)->getIndex() + 1);

            fprintf(elemFile, " %5d %5d     1     1     0 %5d\n", MAT, TYPE, j);
            break;


        default:
            std::cerr << "PhysicalModel::exportPatran : unknown type for cell "<< cell->getIndex()+1 << ", neither HEXAHEDRON, WEDGE, THETRAHEDRON nor QUAD. Cant' export in Patran format." << std::endl;
            continue;
        }

    }

    fclose(elemFile);
}


// ----------------------- exportPatran ------------------
void PhysicalModel::exportPatran(std::string filename) {
    int i;
    unsigned int k;

    // fprintf are used, since this method is copied from an old C routine...
    FILE * outputFile = NULL;
    outputFile = fopen(filename.c_str(),"w");
    if (!outputFile) {
        std::cerr << "Error in PhysicalModel::exportPatran : unable to create output file" << std::endl;
        return;
    }


    int nbPoints = this->getNumberOfAtoms();
    int nbElements = this->getExclusiveComponent(0)->getNumberOfCells();




    //--- patran header -> mostly useless info in our case...
    fprintf(outputFile, "25       0       0       1       0       0       0       0       0\n");
    fprintf(outputFile, "PATRAN File from: %s\n", this->name.c_str());
    fprintf(outputFile, "26       0       0       1   %d    %d       3       4      -1\n", nbPoints, nbElements);
    fprintf(outputFile, "24-Mar-00   05:04:48         3.0\n");


    //--- Nodes (atoms)
    for(i=0; i<nbPoints; i++) {
        SReal pos[3];

        // first line
        // WARNING : indexes are in base 1 !!!!
        fprintf(outputFile, " 1%8d       0       2       0       0       0       0       0\n",
        getAtom(i)->getIndex() + 1);

        // coordinates of this node
        //		fscanf(inputFile, "%d %f %f %f", &j, &x, &y, &z);
        getAtom(i)->getPosition(pos);

        // second line : node coordinates
        fprintf(outputFile, "%16.8E%16.8E%16.8E \n", pos[0], pos[1], pos[2]);

        // third line : ??
        fprintf(outputFile, "1G       6       0       0  000000\n");
    }



    //--- Elements : exlusive cells
    for(i=0; i<nbElements; i++) {
        int typeElement;

        // get the cell
        Cell * cell = this->getExclusiveComponent(0)->getCell(i);

        switch (cell->getType()) {
        case StructureProperties::HEXAHEDRON:
            typeElement = 8;
            break;
        case StructureProperties::WEDGE:
            typeElement = 7;
            break;
        default:
            std::cerr << "PhysicalModel::exportPatran : unknown type for cell "<< cell->getIndex()+1 << ", neither HEXAHEDRON nor WEDGE. Cant' export in Patran format." << std::endl;
            continue;
        }


        //      fscanf(inputFile, "%d 1 %s %d %d %d %d %d %d %d %d",
        //                         &j, &line, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8);

        // first element line
        fprintf(outputFile, " 2%8d%8d       2       0       0       0       0       0\n",
                cell->getIndex() + 1, typeElement);

        // second element line
        fprintf(outputFile, "%8d       0       1       0 0.000000000E+00 0.000000000E+00 0.000000000E+00\n", cell->getNumberOfStructures());

        // third element line : list of nodes
        for (k=0; k<cell->getNumberOfStructures(); k++) {
            fprintf(outputFile, "%8d", cell->getStructure(k)->getIndex() + 1);
        }
        fprintf(outputFile, "\n");

        //    	fprintf(outputFile, "%8d%8d%8d%8d%8d%8d%8d%8d\n",
        //			p1, p2, p3, p4, p5, p6, p7, p8);

    }



    //--- final line
    fprintf(outputFile, "99       0       0       1       0       0       0       0       0\n");


    fclose(outputFile);
}

