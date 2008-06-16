/***************************************************************************
								PMLStiffSpringForceField
                             -------------------
    begin             : September 11th, 2006
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

//-------------------------------------------------------------------------
//						--   Description   --
//	PMLStiffSpringForceField translate an FEM object from Physical Model structure
//  to sofa structure, using StiffSpringForceField.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef PMLSTIFFSPRINGFORCEFIELD_H
#define PMLSTIFFSPRINGFORCEFIELD_H

#include "PMLBody.h"

#include <StructuralComponent.h>
#include "sofa/core/componentmodel/topology/BaseMeshTopology.h"
#include "sofa/component/collision/TriangleModel.h"
//#include "sofa/component/collision/LineModel.h"
//#include "sofa/component/collision/PointModel.h"
#include "sofa/component/forcefield/MeshSpringForceField.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::forcefield;
using namespace std;

class PMLStiffSpringForceField: public PMLBody
{
public :

    PMLStiffSpringForceField(StructuralComponent* body, GNode * parent);

    ~PMLStiffSpringForceField();

    string isTypeOf() { return "StiffSpring"; }

    ///accessors
    TriangleModel * getTriangleModel() { return tmodel; }
    //LineModel * getLineModel() { return lmodel; }
    //PointModel * getPointModel() { return pmodel; }

    ///merge a body with current object
    bool FusionBody(PMLBody*);

    Vector3 getDOF(unsigned int index);
    GNode* getPointsNode() {return parentNode;}

private :

    /// creation of the scene graph
    void createMechanicalState(StructuralComponent* body);
    void createTopology(StructuralComponent* body);
    void createMass(StructuralComponent* body);
    void createVisualModel(StructuralComponent* body);
    void createForceField();
    void createCollisionModel();

    // extract edges to a list of lines
    BaseMeshTopology::Line * hexaToLines(Cell* pCell);
    BaseMeshTopology::Line * tetraToLines(Cell* pCell);
    BaseMeshTopology::Line * triangleToLines(Cell* pCell);
    BaseMeshTopology::Line * quadToLines(Cell* pCell);

    //initialization of properties
    void initMass(string m);
    void initDensity(string m);

    //structure
    MeshSpringForceField<Vec3Types> *Sforcefield;
    TriangleModel * tmodel;
    //LineModel * lmodel;
    //PointModel * pmodel;

    //members for the mass (only one of the 2 vectors is filled)
    std::vector<SReal> massList;
    std::vector<SReal> density;

    //properties
    SReal  ks;			// spring stiffness
    SReal  kd;			// damping factor

};

}
}
}

#endif

