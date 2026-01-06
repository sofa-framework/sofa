/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/collision/geometry/TriangleModelInRegularGrid.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.inl>

namespace sofa::component::collision::geometry
{

using namespace sofa::type;
using namespace sofa::core::topology;
using namespace sofa::defaulttype;

void registerTriangleModelInRegularGrid(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Collision model using a triangular mesh in a regular grid, as described in BaseMeshTopology.")
        .add< TriangleModelInRegularGrid >());
}

TriangleModelInRegularGrid::TriangleModelInRegularGrid() : TriangleCollisionModel<sofa::defaulttype::Vec3Types>()
{

}


TriangleModelInRegularGrid::~TriangleModelInRegularGrid()
{

}


void TriangleModelInRegularGrid::init()
{
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>::init();

    _topology = this->getContext()->getMeshTopology();
    m_mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (!m_mstate) { msg_error() << "TriangleModelInRegularGrid requires a Vec3 Mechanical Model"; return; }
    if (!m_topology) { msg_error() << "TriangleModelInRegularGrid requires a BaseMeshTopology"; return; }

    // Test if _topology depend on an higher topology (to compute Bounding Tree faster) and get it
    TopologicalMapping* _topoMapping = nullptr;
    vector<TopologicalMapping*> topoVec;
    getContext()->get<TopologicalMapping> ( &topoVec, core::objectmodel::BaseContext::SearchRoot );
    _higher_topo = m_topology;
    _higher_mstate = m_mstate;
    bool found = true;
    while ( found )
    {
        found = false;
        for (const auto& v : topoVec)
        {
            if ( v->getTo() == _higher_topo )
            {
                found = true;
                _topoMapping = v;
                _higher_topo = _topoMapping->getFrom();
                if ( !_higher_topo ) break;
                const sofa::simulation::Node* node = static_cast< sofa::simulation::Node* > ( _higher_topo->getContext() );
                _higher_mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > ( node->getMechanicalState() );
            }
        }
    }
    if ( _topoMapping && !_higher_topo ) { 
        msg_error() << "Topological Mapping " << _topoMapping->getName() << " returns a from topology pointer equal to nullptr.";
        return;
    }
    else if (_higher_topo != _topology) {
        msg_info() << "Using the " << _higher_topo->getClassName() << " \"" << _higher_topo->getName() << "\" to compute the bounding trees.";
    }
    else {
        msg_info() << "Keeping the TriangleCollisionModel<sofa::defaulttype::Vec3Types> to compute the bounding trees.";
    }
}

void TriangleModelInRegularGrid::computeBoundingTree ( int )
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    updateFromTopology();
    if ( m_needsUpdate && !cubeModel->empty() ) cubeModel->resize ( 0 );
    if ( !isMoving() && !cubeModel->empty() && !m_needsUpdate ) return; // No need to recompute BBox if immobile

    m_needsUpdate=false;
    Vec3 minElem, maxElem;
    const VecCoord& xHigh =_higher_mstate->read(core::vec_id::read_access::position)->getValue();
    const VecCoord& x =m_mstate->read(core::vec_id::read_access::position)->getValue();

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
            const Vec3& pt1 = xHigh[i];
            if ( pt1[0] > maxElem[0] ) maxElem[0] = pt1[0];
            else if ( pt1[0] < minElem[0] ) minElem[0] = pt1[0];
            if ( pt1[1] > maxElem[1] ) maxElem[1] = pt1[1];
            else if ( pt1[1] < minElem[1] ) minElem[1] = pt1[1];
            if ( pt1[2] > maxElem[2] ) maxElem[2] = pt1[2];
            else if ( pt1[2] < minElem[2] ) minElem[2] = pt1[2];
        }

        for (std::size_t i=0; i<getSize(); ++i)
        {
            Triangle t(this,i);
            const Vec3& pt1 = x[t.p1Index()];
            const Vec3& pt2 = x[t.p2Index()];
            const Vec3& pt3 = x[t.p3Index()];
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();
        }

        cubeModel->setLeafCube ( 0, std::make_pair ( this->begin(),this->end() ), minElem, maxElem ); // define the bounding box of the current triangle
    }
}

}  // namespace sofa::component::collision::geometry
