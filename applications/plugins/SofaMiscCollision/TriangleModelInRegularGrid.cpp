/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaMiscCollision/TriangleModelInRegularGrid.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/TriangleModel.inl>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/gl/template.h>
#include <iostream>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::core::topology;
using namespace sofa::defaulttype;
using helper::vector;

SOFA_DECL_CLASS ( TriangleInRegularGrid )

int TriangleModelInRegularGridClass = core::RegisterObject ( "collision model using a triangular mesh in a regular grid, as described in BaseMeshTopology" )
        .add< TriangleModelInRegularGrid >()
        ;

TriangleModelInRegularGrid::TriangleModelInRegularGrid()
    :TriangleModel()
{

}


TriangleModelInRegularGrid::~TriangleModelInRegularGrid()
{

}


void TriangleModelInRegularGrid::init()
{
    TriangleModel::init();

    _topology = this->getContext()->getMeshTopology();
    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if( !mstate) { serr << "TriangleModelInRegularGrid requires a Vec3 Mechanical Model" << sendl; return;}
    if (!_topology) { serr << "TriangleModelInRegularGrid requires a BaseMeshTopology" << sendl; return;}

    // Test if _topology depend on an higher topology (to compute Bounding Tree faster) and get it
    TopologicalMapping* _topoMapping = NULL;
    vector<TopologicalMapping*> topoVec;
    getContext()->get<TopologicalMapping> ( &topoVec, core::objectmodel::BaseContext::SearchRoot );
    _higher_topo = _topology;
    _higher_mstate = mstate;
    bool found = true;
    while ( found )
    {
        found = false;
        for ( vector<TopologicalMapping*>::iterator it = topoVec.begin(); it != topoVec.end(); ++it )
        {
            if ( ( *it )->getTo() == _higher_topo )
            {
                found = true;
                _topoMapping = *it;
                _higher_topo = _topoMapping->getFrom();
                if ( !_higher_topo ) break;
                sofa::simulation::Node* node = static_cast< sofa::simulation::Node* > ( _higher_topo->getContext() );
                _higher_mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > ( node->getMechanicalState() );
            }
        }
    }
    if ( _topoMapping && !_higher_topo ) { serr << "Topological Mapping " << _topoMapping->getName() << " returns a from topology pointer equal to NULL." << sendl; return;}
    else if ( _higher_topo != _topology ) sout << "Using the " << _higher_topo->getClassName() << " \"" << _higher_topo->getName() << "\" to compute the bounding trees." << sendl;
    else sout << "Keeping the TriangleModel to compute the bounding trees." << sendl;
}

void TriangleModelInRegularGrid::computeBoundingTree ( int )
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if ( needsUpdate && !cubeModel->empty() ) cubeModel->resize ( 0 );
    if ( !isMoving() && !cubeModel->empty() && !needsUpdate ) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;
    const VecCoord& xHigh =_higher_mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();

    // no hierarchy
    if ( empty() )
        cubeModel->resize ( 0 );
    else
    {
        cubeModel->resize ( 1 );
        minElem = xHigh[0];
        maxElem = xHigh[0];
        for ( unsigned i=1; i<xHigh.size(); ++i )
        {
            const Vector3& pt1 = xHigh[i];
            if ( pt1[0] > maxElem[0] ) maxElem[0] = pt1[0];
            else if ( pt1[0] < minElem[0] ) minElem[0] = pt1[0];
            if ( pt1[1] > maxElem[1] ) maxElem[1] = pt1[1];
            else if ( pt1[1] < minElem[1] ) minElem[1] = pt1[1];
            if ( pt1[2] > maxElem[2] ) maxElem[2] = pt1[2];
            else if ( pt1[2] < minElem[2] ) minElem[2] = pt1[2];
        }

        for (int i=0; i<getSize(); ++i)
        {
            Triangle t(this,i);
            const Vector3& pt1 = x[t.p1Index()];
            const Vector3& pt2 = x[t.p2Index()];
            const Vector3& pt3 = x[t.p3Index()];
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();
        }

        cubeModel->setLeafCube ( 0, std::make_pair ( this->begin(),this->end() ), minElem, maxElem ); // define the bounding box of the current triangle
    }
}

}

}

}
