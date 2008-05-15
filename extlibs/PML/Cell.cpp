/***************************************************************************
                                  Cell.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.9 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "CellProperties.h"
#include "Cell.h"
#include "PhysicalModel.h"

#include <math.h>

//----------------------- Constructors -----------------------
Cell::Cell(PhysicalModel *p, const StructureProperties::GeometricType t)
     : StructuralComponent(p) {
    this->Structure::properties = new CellProperties(p,t);
}

Cell::Cell(PhysicalModel *p, const StructureProperties::GeometricType t, xmlNodePtr node)
     : StructuralComponent(p, node) {
	for (xmlNodePtr child = node->xmlChildrenNode; child != NULL; child = child->next){
		//find properties node
		if (!xmlStrcmp(child->name,(const xmlChar*)"cellProperties"))
		this->Structure::properties = new CellProperties(p,t, child);
		//find nrOfStructures node
		if (!xmlStrcmp(child->name,(const xmlChar*)"nrOfStructures")) {
			xmlChar * prop = xmlGetProp(child, (xmlChar*)("value"));
			unsigned int val = atoi((char*)prop);
			this->plannedNumberOfStructures(val);
		}
	}
}

Cell::Cell(PhysicalModel *p, const StructureProperties::GeometricType t, const unsigned int ind)
     : StructuralComponent(p) {
    this->Structure::properties = new CellProperties(p, t, ind);
}

//----------------------- Destructor -----------------------
Cell::~Cell() {
    // delete the structural component properties
    deleteProperties();

    // if (Structure::properties)
    delete (CellProperties*) Structure::properties;

    if (StructuralComponent::atomList)
        delete StructuralComponent::atomList;
    StructuralComponent::atomList = NULL;

    // delete all children
    deleteAllStructures();

    // tell all parents that I am going away to the paradise of pointers
    removeFromParents();
}

//----------------------- setIndex -----------------------
bool Cell::setIndex(const unsigned int index) {
    // set the property
    Structure::setIndex(index);
    // tell the physical model about the change (and return true if insertion was ok)
    return getPhysicalModel()->addGlobalIndexCellPair(std::GlobalIndexStructurePair(index,this));
}

// --------------- makePrintData ---------------
bool Cell::makePrintData(const StructuralComponent * sc) {
    // if the sc (=the object that is calling this method) is the first in the
    // mySC list to be a exclusive component, it means the cell has not been written yet. If it is not
    // the first exclusive component in the mySC list, the cell could simply writes its cell ref
    bool isExclusive = false;
    // search the first exclusive
    unsigned int i=0;
    while (!isExclusive && i<getNumberOfStructuralComponents()) {
        isExclusive = getStructuralComponent(i)->isExclusive();
        if (!isExclusive)
            i++;
    }
    // post condition here:
    // - either isExclusive && i is the index of the first exclusive
    // - or !isExclusive && there are no exclusive sc containing this cell

    // print the cell data (and not the cellRef) only if :
    // - the first exclusive component is sc
    // - there are no exclusive components for this cell and sc is the first SC
    if (isExclusive)
        return (sc==getStructuralComponent(i));
    else
        return (sc==getStructuralComponent(0));
}

// --------------- xmlPrint ---------------
void Cell::xmlPrint(std::ostream &o, const StructuralComponent *sc) {

    // do what's our duty
    if (makePrintData(sc)) {
        // print out all the information...
        o << "<cell>" << std::endl;

        // ...the properties...
        ((CellProperties *)Structure::properties)->xmlPrint(o);

        // ... and the color (if it is not the default one) ...
        if (((StructuralComponentProperties *)StructuralComponent::properties)->getColor() != StructuralComponentProperties::DEFAULT) {
            o << "<color r=\"" << ((StructuralComponentProperties *)StructuralComponent::properties)->getRed();
            o << "\" g=\"" << ((StructuralComponentProperties *)StructuralComponent::properties)->getGreen();
            o << "\" b=\"" << ((StructuralComponentProperties *)StructuralComponent::properties)->getBlue();
            o << "\" a=\"" << ((StructuralComponentProperties *)StructuralComponent::properties)->getAlpha() << "\" />" << std::endl;
        }

        // optimize the memory allocation for reading
        o << "<nrOfStructures value=\"" << structures.size() << "\"/>" << std::endl;

        // ... and finally the ref to the atoms
        for (unsigned int i=0; i<structures.size(); i++) {
            o << "<atomRef index=\"" << structures[i]->getIndex() << "\" />" << std::endl;
        }
        o << "</cell>" << std::endl;
    } else {
        // print only out the cellRef
        o << "<cellRef index=\"" << getIndex() << "\" />" << std::endl;
    }
}

// ------------------ deleteAllStructures ---------------------
void Cell::deleteAllStructures() {
    // here the list elements (i.e. atoms) are NOT deleted
    // but the list is cleared
    structures.clear();
}

// ------------------ getProperties ---------------------
CellProperties * Cell::getProperties() {
    return ((CellProperties *)Structure::properties);
}

// ------------------ getFacets ---------------------
StructuralComponent * Cell::getFacets() {

	StructuralComponent * facets = new StructuralComponent(getPhysicalModel());
	Cell * face;
	switch (getProperties()->getType())
	{
		case StructureProperties::HEXAHEDRON :
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(3));
			face->addStructure(getStructure(2)); face->addStructure(getStructure(1));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(4));
			face->addStructure(getStructure(7)); face->addStructure(getStructure(3));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(1));
			face->addStructure(getStructure(5)); face->addStructure(getStructure(4));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(3)); face->addStructure(getStructure(7));
			face->addStructure(getStructure(6)); face->addStructure(getStructure(2));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(1)); face->addStructure(getStructure(2));
			face->addStructure(getStructure(6)); face->addStructure(getStructure(5));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(4)); face->addStructure(getStructure(5));
			face->addStructure(getStructure(6)); face->addStructure(getStructure(7));
			facets->addStructure(face);
			return facets;
		case StructureProperties::TETRAHEDRON :
			face = new Cell(getPhysicalModel(), StructureProperties::TRIANGLE);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(1));
			face->addStructure(getStructure(2));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::TRIANGLE);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(2));
			face->addStructure(getStructure(3));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::TRIANGLE);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(3));
			face->addStructure(getStructure(1));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::TRIANGLE);
			face->addStructure(getStructure(2)); face->addStructure(getStructure(1));
			face->addStructure(getStructure(3));
			facets->addStructure(face);
			return facets;
		case StructureProperties::WEDGE :
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(2)); face->addStructure(getStructure(5));
			face->addStructure(getStructure(4)); face->addStructure(getStructure(1));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(0)); face->addStructure(getStructure(1));
			face->addStructure(getStructure(4)); face->addStructure(getStructure(3));
			facets->addStructure(face);
			face = new Cell(getPhysicalModel(), StructureProperties::QUAD);
			face->addStructure(getStructure(2)); face->addStructure(getStructure(0));
			face->addStructure(getStructure(3)); face->addStructure(getStructure(5));
			facets->addStructure(face);
			return facets;
		case StructureProperties::QUAD :
		case StructureProperties::TRIANGLE :
			facets->addStructure(this);
			return facets;
		default :
			return NULL;
	}

}

// ------------------ normal ---------------------
SReal * Cell::normal() 
{
	if (getProperties()->getType()== StructureProperties::QUAD || getProperties()->getType()== StructureProperties::TRIANGLE)
	{
		SReal * N = new SReal[3];
		SReal v1[3], v2[3];
		SReal posi[3], posip1[3];
		((Atom*)getStructure(0))->getPosition(posip1);
		((Atom*)getStructure(2))->getPosition(posi);
		v1[0] = posi[0]-posip1[0]; v1[1] = posi[1]-posip1[1]; v1[2] = posi[2]-posip1[2]; 
		((Atom*)getStructure(1))->getPosition(posi);
		v2[0] = posi[0]-posip1[0]; v2[1] = posi[1]-posip1[1]; v2[2] = posi[2]-posip1[2]; 
			
		N[0] = v1[1]*v2[2] - v1[2]*v2[1];
		N[1] = v1[2]*v2[0] - v1[0]*v2[2];
		N[2] = v1[0]*v2[1] - v1[1]*v2[0]; 

		SReal norm = sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);

		N[0] /= norm; 
		N[1] /= norm; 
		N[2] /= norm;
		return N;
	}
	return NULL;
	
}

// ------------------ surface ---------------------
SReal Cell::surface() 
{
	if (getProperties()->getType()== StructureProperties::QUAD || getProperties()->getType()== StructureProperties::TRIANGLE)
	{
		SReal A[3]={0.0, 0.0, 0.0 };
		unsigned int nbElem;

		nbElem = Cell::getNumberOfStructures();    

		SReal posi[3], posip1[3];
		((Atom*)getStructure(0))->getPosition(posip1);
	    
		for (unsigned int i=0;i<nbElem;i++) {
			posi[0]=posip1[0]; posi[1]=posip1[1]; posi[2]=posip1[2]; 
			((Atom*)getStructure((i+1)%nbElem))->getPosition(posip1);
			//Cross(posi, posip1, inter);
			A[0] += posi[1]*posip1[2] - posi[2]*posip1[1];
			A[1] += posi[2]*posip1[0] - posi[0]*posip1[2];
			A[2] += posi[0]*posip1[1] - posi[1]*posip1[0];
		}
	    
		// here A is in fact twice the theoritical area vector
		A[0] /= 2.0; A[1] /= 2.0; A[2] /= 2.0;

		// face normal :
		SReal * N = normal();

		SReal surface = N[0]*A[0] + N[1]*A[1] + N[2]*A[2];

		return surface>0?surface:-surface;
	}
	else {
		StructuralComponent * facets = getFacets();
		if (!facets)
			return 0.0;
		SReal surface=0.0;
		for (unsigned int i=0 ; i<facets->getNumberOfCells() ; i++) {
			surface += facets->getCell(i)->surface();
		}
		return surface;
	}
}


// ------------------ volume ---------------------
SReal Cell::volume() 
{
	StructuralComponent * facets = getFacets();
	if (!facets || getProperties()->getType()== StructureProperties::QUAD || getProperties()->getType()== StructureProperties::TRIANGLE)
		return 0.0;

	SReal vol=0.0;
	Cell * face;
	for (unsigned int i=0;i < facets->getNumberOfCells();i++) {
		face = facets->getCell(i);
		SReal pos[3];
		((Atom*)face->getStructure(0))->getPosition(pos);
		SReal * N = face->normal();
		vol += face->surface()* ( pos[0]*N[0] + pos[1]*N[1] + pos[2]*N[2] );
	}

	vol /= 3.0;
	return vol>0?vol:-vol;

}
