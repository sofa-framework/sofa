
#include "PMLTransform.h"


#include "MultiComponent.h"
#include "StructuralComponent.h"
#include "Atom.h"
#include <CellProperties.h>
#include <AtomProperties.h>
#include <StructuralComponentProperties.h>

#include <iostream>
#include <string>

using namespace std;


// -------------------------  FACET METHODS  ------------------------//

// -------------------- constructor/destructor ------------------------
Facet::Facet(unsigned int size, unsigned int id[]) {
    this->size = size;
    this->id = new unsigned int[size];
    for (unsigned int i=0;i<size; i++)
        this->id[i]=id[i];
    used = 1;
}

Facet::~Facet() {
    delete [] id;
}

// -------------------- debug ------------------------
void Facet::debug() {
    switch (size) {
        case 3:
            cout << "triangle <";
            break;
        case 4:
            cout << "quad <";
            break;
        default:
            cout << "unknown facet <";
            break;
    }
    unsigned int i;
    for (i=0;i<size-1;i++)
        cout << id[i] << ",";
    cout << id[i] << "> used " << used << " times" << endl;
}

// -------------------- testEquivalence ------------------------
bool Facet::testEquivalence(unsigned int size, unsigned int id[]) {
    if (this->size != size) {
        return false;
    }
    else {
        unsigned int i=0;
        while (i<size && isIn(id[i])) {
            i++;
        }
    
        if (i==size)
            used++;
    
        return (i==size);
    }
}

// -------------------- isIn ------------------------
bool Facet::isIn(unsigned int index) const {
    unsigned int i=0;
    while (i<size && id[i]!=index)
        i++;
    return (i!=size);
}

// -------------------- getCell ------------------------
Cell * Facet::getCell(PhysicalModel *pm) const {
    Cell *c;
    // create the correct geometric type cell
    switch (size) {
        case 3:
            c = new Cell(NULL, StructureProperties::TRIANGLE);
            break;
        case 4:
            c = new Cell(NULL, StructureProperties::QUAD);
            break;
        default:
            c=NULL;
    }
    // get the atom corresponding to the index stored in id
    // and insert them in the cell
    for (unsigned int i=0;i<size; i++) {
        Atom *a = pm->getAtom(id[i]);
        if (a==NULL) {
            cout << "Argh! Cannot find atom #" << id[i] << endl;
        }
        else {
            c->addStructureIfNotIn(a);
        }
    }
    return c;
}

// -------------------- getUsed ------------------------
unsigned int Facet::getUsed() const {
    return used;
}



///// ------------------ PMLTRANSFORM METHODS ------------//
//--------   elements to neighborhood methods  -----------//

std::map<unsigned int, Cell*> PMLTransform::neighMap;
std::vector <Facet *> PMLTransform::allFacets;

// -------------------- getIterator ------------------------
// get the iterator on the correct atom index in the neighMap
// if non existant create it
map<unsigned int, Cell*>::iterator PMLTransform::getIterator(unsigned int index) {

    map<unsigned int, Cell*>::iterator it;
    
    // find atom index in the map
    it = neighMap.find(index);
    
    // not present, insert a new cell, associated with index
    if (it==neighMap.end()) {
        // instanciate
        Cell *nc = new Cell(NULL, StructureProperties::POLY_VERTEX);
        // name
        stringstream n(std::stringstream::out);
        n << "Atom #" << index << '\0';
        ((Structure *)nc)->setName(n.str());
        // insert the new cell in the map
        pair<map<unsigned int, Cell*>::iterator, bool> ins = neighMap.insert(map<unsigned int, Cell*>::value_type(index, nc));
        // it show where it has been inserted
        it = ins.first;
    }
    
    return it;
}


// -------------------- generateNeighborhood ------------------------
/// generate the neighborhoods
StructuralComponent * PMLTransform::generateNeighborhood(StructuralComponent *sc) {

	PMLTransform::neighMap.clear();
    // an iterator
    map<unsigned int, Cell*>::iterator it;
    
    // for each cells recreate neighborhoods
    for (unsigned int i=0; i<sc->getNumberOfCells(); i++) {
        // c is an hexahedron
        Cell *c = sc->getCell(i);

        switch(c->getType()) {
            case StructureProperties::WEDGE:
                if (c->getNumberOfStructures()!=6) {
                    cout << "cell #" << c->getIndex() << " of type TETRAHEDRON does not contains 4 atoms (found only " <<  c->getNumberOfStructures() << ". Cell skipped." << endl;
                }
                else {
                    // WEDGE
                    //     1-------------4       facets (quad):   facets (triangles):     lines:
                    //     /\           . \       2,5,4,1          0,2,1                   0,1      2,5
                    //    /  \         /   \      0,1,4,3          3,4,5                   0,2      3,4
                    //   0- - \ - - - 3     \     2,0,3,5                                  1,2      4,5
                    //     \   \         \   \                                             0,3      5,3
                    //       \ 2-----------\--5                                            1,4
                    it = getIterator(c->getStructure(0)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(1));
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                    (*it).second->addStructureIfNotIn(c->getStructure(3));

                    it = getIterator(c->getStructure(1)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(0));
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                    (*it).second->addStructureIfNotIn(c->getStructure(4));

                    it = getIterator(c->getStructure(2)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(0));
                    (*it).second->addStructureIfNotIn(c->getStructure(1));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));

                    it = getIterator(c->getStructure(3)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(4));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));
                    (*it).second->addStructureIfNotIn(c->getStructure(0));

                    it = getIterator(c->getStructure(4)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(3));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));
                    (*it).second->addStructureIfNotIn(c->getStructure(1));

                    it = getIterator(c->getStructure(5)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(3));
                    (*it).second->addStructureIfNotIn(c->getStructure(4));
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                }
                break;

            case StructureProperties::TETRAHEDRON:
                if (c->getNumberOfStructures()!=4) {
                    cout << "cell #" << c->getIndex() << " of type TETRAHEDRON does not contains 4 atoms (found only " <<  c->getNumberOfStructures() << ". Cell skipped." << endl;
                }
                else {
                    // tetrahedron are defined as follow:
                    //                    3                   triangular base: 0,1,2
                    //                  /| \                          -
                    //                 / |  \                 So to generate the neighborhodd,
                    //                2__|___\1               we just have to loop on all the
                    //                \  |   /                atoms and add them their corresponding neigh
                    //                 \ |  /                 This is easy as in a tetrahedron all atoms
                    //                  \|/                   are neighbors to all atoms...
                    //                  0
                    for (unsigned int j=0; j<4;j++) {
                        // get current atom
                        it = getIterator(c->getStructure(j)->getIndex());
                        // add all others to its neighborhood
                        (*it).second->addStructureIfNotIn(c->getStructure((j+1)%4));
                        (*it).second->addStructureIfNotIn(c->getStructure((j+2)%4));
                        (*it).second->addStructureIfNotIn(c->getStructure((j+3)%4));
                    }
                }
                break;
            
            case StructureProperties::HEXAHEDRON:
                if (c->getNumberOfStructures()!=8) {
                    cout << "cell #" << c->getIndex() << " of type HEXAHEDRON does not contains 8 atoms (found only " <<  c->getNumberOfStructures() << ". Cell skiped." << endl;
                }
                else {
                    // hexahedron are defined as follow:
                    //              2-------------6
                    //             / \           / \      So to generate the neighborhood,
                    //            /   \         /   \     we just have to loop on all the
                    //           1-----\-------5     \    atoms and add them their corresponding neigh
                    //           \     3-------------7
                    //            \   /         \   /
                    //             \ /           \ /
                    //              0-------------4
                            
                    // atom 0 is neigbor of atoms : 1, 3, 4
                    it = getIterator(c->getStructure(0)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(1));
                    (*it).second->addStructureIfNotIn(c->getStructure(3));
                    (*it).second->addStructureIfNotIn(c->getStructure(4));
                    
                    // atom 1 is neigbor of atoms : 0, 2, 5
                    it = getIterator(c->getStructure(1)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(0));
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));
                    
                    // atom 2 is neigbor of atoms : 1, 3, 6
                    it = getIterator(c->getStructure(2)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(1));
                    (*it).second->addStructureIfNotIn(c->getStructure(3));
                    (*it).second->addStructureIfNotIn(c->getStructure(6));
                    
                    // atom 3 is neigbor of atoms : 0, 2, 7
                    it = getIterator(c->getStructure(3)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(0));
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                    (*it).second->addStructureIfNotIn(c->getStructure(7));
                    
                    // atom 4 is neigbor of atoms : 0, 5, 7
                    it = getIterator(c->getStructure(4)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(0));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));
                    (*it).second->addStructureIfNotIn(c->getStructure(7));
                    
                    // atom 5 is neigbor of atoms : 1, 4, 6
                    it = getIterator(c->getStructure(5)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(1));
                    (*it).second->addStructureIfNotIn(c->getStructure(4));
                    (*it).second->addStructureIfNotIn(c->getStructure(6));
                    
                    // atom 6 is neigbor of atoms : 2, 5, 7
                    it = getIterator(c->getStructure(6)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(2));
                    (*it).second->addStructureIfNotIn(c->getStructure(5));
                    (*it).second->addStructureIfNotIn(c->getStructure(7));
                    
                    // atom 7 is neigbor of atoms : 3, 4, 6
                    it = getIterator(c->getStructure(7)->getIndex());
                    (*it).second->addStructureIfNotIn(c->getStructure(3));
                    (*it).second->addStructureIfNotIn(c->getStructure(4));
                    (*it).second->addStructureIfNotIn(c->getStructure(6));
                }
                break;
            default:
                cout << "Cannot translate cell #" << c->getIndex() << ": it is not a HEXAHEDRON nor a TETRAHEDRON." << endl;
                break;
        }
    }
    
    // now that we have in neighMap all the neighborhoods, just add them in a new SC
    StructuralComponent *neigh = new StructuralComponent(NULL, "Neighborhoods");
    
    for (it=neighMap.begin(); it!=neighMap.end(); it++) {
        neigh->addStructure((*it).second);
    }
    
    return neigh;
}


// -------------------- equivalent ------------------------
// check if equivalent of already existing facet
void PMLTransform::equivalent(int size, unsigned int id[]) {
    vector <Facet *>::iterator it;
 
    // look into allFacets for equivalence   
    it = allFacets.begin();
    while (it!=allFacets.end() && !(*it)->testEquivalence(size, id))
        it++;
    
    // not found => insert
    if (it==allFacets.end())
        allFacets.push_back(new Facet(size,id));
}


// -------------------- generateExternalSurface ------------------------
/// generate the outside surface
MultiComponent * PMLTransform::generateExternalSurface(StructuralComponent *sc) {
	PMLTransform::allFacets.clear();
    // outside/external facets are facets that are used in only one element

    // for each cells update the counter
    for (unsigned int i=0; i<sc->getNumberOfCells(); i++) {
        // c is an hexahedron
        Cell *c = sc->getCell(i);

        // Facets have to be described in anticlockwise (trigonometrywise) when
        // looking at them from outside        
        switch(c->getType()) {
            case StructureProperties::WEDGE: {
                    // WEDGE
                    //     1-------------4       facets (quad):   facets (triangles):     lines:
                    //     /\           . \       2,5,4,1          0,2,1                   0,1      2,5
                    //    /  \         /   \      0,1,4,3          3,4,5                   0,2      3,4
                    //   0- - \ - - - 3     \     2,0,3,5                                  1,2      4,5
                    //     \   \         \   \                                             0,3      5,3
                    //       \ 2-----------\--5                                            1,4
                    unsigned int idQ[4];
                    idQ[0]=c->getStructure(2)->getIndex();
                    idQ[1]=c->getStructure(5)->getIndex();
                    idQ[2]=c->getStructure(4)->getIndex();
                    idQ[3]=c->getStructure(1)->getIndex();
                    equivalent(4,idQ);

                    idQ[0]=c->getStructure(0)->getIndex();
                    idQ[1]=c->getStructure(1)->getIndex();
                    idQ[2]=c->getStructure(4)->getIndex();
                    idQ[3]=c->getStructure(3)->getIndex();
                    equivalent(4,idQ);

                    idQ[0]=c->getStructure(2)->getIndex();
                    idQ[1]=c->getStructure(0)->getIndex();
                    idQ[2]=c->getStructure(3)->getIndex();
                    idQ[3]=c->getStructure(5)->getIndex();
                    equivalent(4,idQ);

                    unsigned int idT[3];
                    idT[0]=c->getStructure(0)->getIndex();
                    idT[1]=c->getStructure(2)->getIndex();
                    idT[2]=c->getStructure(1)->getIndex();
                    equivalent(3,idT);

                    idT[0]=c->getStructure(3)->getIndex();
                    idT[1]=c->getStructure(4)->getIndex();
                    idT[2]=c->getStructure(5)->getIndex();
                    equivalent(3,idT);
            }
            case StructureProperties::TETRAHEDRON:
                {
                    // tetrahedron are defined as follow:
                    //                    3
                    //                  /| \                  So to generate the facet list, 
                    //                 / |  \                 we just have to loop on all the
                    //                1__|___\ 2              tetrahedron and add the corresponding 4 facets :
                    //                \  |   /                f0=0,1,2      f2=0,3,1 
                    //                 \ |  /                 f1=0,2,3      f3=2,1,3
                    //                  \|/
                    //                  0
                    break;
                    unsigned int id[3];        
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(1)->getIndex();
                    id[2]=c->getStructure(2)->getIndex();
                    equivalent(3,id);
                    
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(2)->getIndex();
                    id[2]=c->getStructure(3)->getIndex();
                    equivalent(3,id);
                    
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(3)->getIndex();
                    id[2]=c->getStructure(1)->getIndex();
                    equivalent(3,id);
                    
                    id[0]=c->getStructure(2)->getIndex();
                    id[1]=c->getStructure(1)->getIndex();
                    id[2]=c->getStructure(3)->getIndex();
                    equivalent(3,id);
                }
                
            case StructureProperties::HEXAHEDRON: 
                {
                    // hexahedron are defined as follow:
                    //              2-------------6
                    //             / \           . \      So to generate the facet list, 
                    //            /   \         /   \     we just have to loop on all the
                    //           1- - -\ - - - 5     \    hexahedron and add the corresponding 6 facets :
                    //           \     3-------------7    f0=0,3,2,1     f3=3,7,6,2
                    //            \   /         \   /     f1=0,4,7,3     f4=1,2,6,5
                    //             \ /           . /      f2=0,1,5,4     f5=4,5,6,7
                    //              0-------------4
            
                    unsigned int id[4];        
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(3)->getIndex();
                    id[2]=c->getStructure(2)->getIndex();
                    id[3]=c->getStructure(1)->getIndex();
                    equivalent(4,id);
                    
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(4)->getIndex();
                    id[2]=c->getStructure(7)->getIndex();
                    id[3]=c->getStructure(3)->getIndex();
                    equivalent(4,id);
                    
                    id[0]=c->getStructure(0)->getIndex();
                    id[1]=c->getStructure(1)->getIndex();
                    id[2]=c->getStructure(5)->getIndex();
                    id[3]=c->getStructure(4)->getIndex();
                    equivalent(4,id);
                    
                    id[0]=c->getStructure(3)->getIndex();
                    id[1]=c->getStructure(7)->getIndex();
                    id[2]=c->getStructure(6)->getIndex();
                    id[3]=c->getStructure(2)->getIndex();
                    equivalent(4,id);
                    
                    id[0]=c->getStructure(1)->getIndex();
                    id[1]=c->getStructure(2)->getIndex();
                    id[2]=c->getStructure(6)->getIndex();
                    id[3]=c->getStructure(5)->getIndex();
                    equivalent(4,id);
                
                    id[0]=c->getStructure(4)->getIndex();
                    id[1]=c->getStructure(5)->getIndex();
                    id[2]=c->getStructure(6)->getIndex();
                    id[3]=c->getStructure(7)->getIndex();
                    equivalent(4,id);
                    break;
                }
            default:
            break;
        }
    }
    
    // now that we have in facetMap all the facet and the number of times they have been used
    // just add in a new SC all the one that are used only once
    StructuralComponent *facet = new StructuralComponent(NULL, "extern");
        
    for (vector <Facet *>::iterator it=allFacets.begin(); it!=allFacets.end(); it++) {
        //(*it)->debug();
        if ((*it)->getUsed()==1) // used only once
            facet->addStructure((*it)->getCell(sc->getPhysicalModel()));
    }
    
    // delete all facets
    for (vector <Facet *>::iterator it=allFacets.begin(); it!=allFacets.end(); it++)
            delete (*it);
            
    // create the enclosed volume
    MultiComponent *mc = new MultiComponent(NULL, "Enclosed Volumes");
    mc->addSubComponent(facet);
    
    // return the SC
    return mc;
}
