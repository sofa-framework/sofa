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

//-------------------------------------------------------------------------
//						--   Description   --
//	PMLStiffSpringForceField translate an FEM object from Physical Model structure
//  to sofa structure, using StiffSpringForceField.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef SOFAPML_PMLSTIFFSPRINGFORCEFIELD_H
#define SOFAPML_PMLSTIFFSPRINGFORCEFIELD_H

#include "PMLBody.h"
#include "initSofaPML.h"

#include <StructuralComponent.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaDeformable/MeshSpringForceField.h>


#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::interactionforcefield;
using namespace std;

class SOFA_BUILD_FILEMANAGER_PML_API PMLStiffSpringForceField: public PMLBody
{
public :

    PMLStiffSpringForceField(StructuralComponent* body, GNode * parent);

    ~PMLStiffSpringForceField();

    string isTypeOf() { return "StiffSpring"; }

    ///accessors
    TriangleModel::SPtr getTriangleModel() { return tmodel; }
    //LineModel * getLineModel() { return lmodel; }
    //PointModel * getPointModel() { return pmodel; }

    ///merge a body with current object
    bool FusionBody(PMLBody*);

    Vector3 getDOF(unsigned int index);
    GNode::SPtr getPointsNode() {return parentNode;}

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
    MeshSpringForceField<Vec3Types>::SPtr Sforcefield;
    TriangleModel::SPtr tmodel;
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

