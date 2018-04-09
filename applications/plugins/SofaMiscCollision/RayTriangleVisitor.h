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
#ifndef SOFA_SIMULATION_RayTriangleVisitor_H
#define SOFA_SIMULATION_RayTriangleVisitor_H
#include "config.h"

#include <sofa/simulation/Visitor.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaOpenglVisual/OglModel.h>
#include <sofa/defaulttype/Vec.h>


namespace sofa
{


namespace component {
namespace collision {


/** Detects intersections of a ray with the TriangleModels and the OglModels.
 * The intersections behind the starting point are not reported.
 * We are only interested in counting the intersections, not modeling their geometry.
 * @author Francois Faure, 2014
 */
class  SOFA_MISC_COLLISION_API  RayTriangleVisitor : public simulation::Visitor
{

public:

    typedef defaulttype::Vec<3,SReal> Vec3;
    defaulttype::Vec3d origin;    ///< Ray starting point
    defaulttype::Vec3d direction; ///< Ray direction

    /// Return the embedding model. In case of nested hierarchy, return the smallest (deepest). NULL if no embedding model.
    core::objectmodel::BaseObject* embeddingModel();

    // generic
    RayTriangleVisitor(const core::ExecParams* params = core::ExecParams::defaultInstance());
    virtual void processTriangleModel(simulation::Node* node, component::collision::TriangleModel* obj);
    virtual void processOglModel(simulation::Node* node, component::visualmodel::OglModel* obj);
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual bool isThreadSafe() const { return true; }
    virtual const char* getCategoryName() const { return "animate"; }
    virtual const char* getClassName() const { return "RayTriangleVisitor"; }

private:
    /// Ray-triangle intersection report.
    struct Hit{
        core::objectmodel::BaseObject* hitObject; ///< the object hit
        SReal distance;                 ///< distance from the ray origin to the hit point
        bool internal;                  ///< true if hit from the inside


    };

    friend struct distanceHitSort;

    helper::vector<Hit> hits;  ///< raw result

};

}
}
} // namespace sofa

#endif
