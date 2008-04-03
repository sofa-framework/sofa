/***************************************************************************
								PMLFemForceField
                             -------------------
    begin             : August 21th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/02/25 13:51:44 $
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
//	PMLFemForceField translate an FEM object from Physical Model structure
//  to sofa structure, using TetrahedronFEMForcefield.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef PMLFEMFORCEFIELD_H
#define PMLFEMFORCEFIELD_H

#include "PMLBody.h"

#include <StructuralComponent.h>
#include "sofa/component/topology/MeshTopology.h"
#include "sofa/component/collision/TriangleModel.h"
//#include "sofa/component/collision/LineModel.h"
//#include "sofa/component/collision/PointModel.h"



#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace std;

class PMLFemForceField: public PMLBody
{
public :

    PMLFemForceField(StructuralComponent* body, GNode * parent);

    ~PMLFemForceField();

    string isTypeOf() { return "FEM"; }

    ///accessors
    TriangleModel * getTriangleModel() { return tmodel; }
    //LineModel * getLineModel() { return lmodel; }
    //PointModel * getPointModel() { return pmodel; }

    ///merge a body with current object
    bool FusionBody(PMLBody*);

    Vec3d getDOF(unsigned int index);

    GNode* getPointsNode() {return parentNode;}

private :

    /// creation of the scene graph
    void createMechanicalState(StructuralComponent* body);
    void createTopology(StructuralComponent* body);
    void createMass(StructuralComponent* body);
    void createVisualModel(StructuralComponent* body);
    void createForceField();
    void createCollisionModel();

    //initialization of properties
    void initMass(string m);
    void initDensity(string m);

    //tesselation of hexahedron to 5 tetrahedrons
    MeshTopology::Tetra * Tesselate(Cell* pCell);

    //structure
    TriangleModel * tmodel;
    //LineModel * lmodel;
    //PointModel * pmodel;

    //members for the mass (only one of the 2 vectors is filled)
    std::vector<double> massList;
    std::vector<double> density;

    //members for FEM properties
    double young;
    double poisson;
    std::string deformationType;

};

}
}
}

#endif

