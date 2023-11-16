/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/gui/common/config.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/fwd.h>
#include <sofa/gui/component/performer/MouseInteractor.h>

namespace sofa::gui::common
{

void SOFA_GUI_COMMON_API decodeCollisionElement( const type::RGBAColor& colour, sofa::gui::component::performer::BodyPicked& body );
void SOFA_GUI_COMMON_API decodePosition( sofa::gui::component::performer::BodyPicked& body, const type::RGBAColor& colour, const sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index);
void SOFA_GUI_COMMON_API decodePosition( sofa::gui::component::performer::BodyPicked& body, const type::RGBAColor& colour, const sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index);

// compat
SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
void SOFA_GUI_COMMON_API decodeCollisionElement( const sofa::type::Vec4f& colour, sofa::gui::component::performer::BodyPicked& body );
SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
void SOFA_GUI_COMMON_API decodePosition( sofa::gui::component::performer::BodyPicked& body, const sofa::type::Vec4f& colour, const sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index);
SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
void SOFA_GUI_COMMON_API decodePosition( sofa::gui::component::performer::BodyPicked& body, const sofa::type::Vec4f& colour, const sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index);


/* Launches the drawColourPicking() method of each CollisionModel */
class SOFA_GUI_COMMON_API ColourPickingVisitor : public simulation::Visitor
{

public:

    enum ColourCode
    {
        ENCODE_COLLISIONELEMENT,		///< The object colour encodes the pair CollisionModel - CollisionElement
        ENCODE_RELATIVEPOSITION,	///< The object colour encodes the relative position.
    };


    /// Picking related. Render the collision model with an appropriate RGB colour code
    /// so as to recognize it with the PickHandler of the GUI.
    /// ENCODE_COLLISIONELEMENT Pass :
    ///   r channel : indexCollisionModel / totalCollisionModelInScene.
    ///   g channel : index of CollisionElement.
    /// ENCODE_RELATIVEPOSITION Pass :
    /// r,g,b channels encode the barycentric weights for a triangle model
    virtual void drawColourPicking(const ColourCode /* method */) {}

    /// Picking related.
    /// For TriangleModels a,b,c encode the barycentric weights with respect to the vertex p1 p2 and p3 of
    /// the TriangleElement with the given index

    ColourPickingVisitor(const core::visual::VisualParams* params, ColourCode Method)
        :simulation::Visitor(sofa::core::visual::visualparams::castToExecParams(params)),vparams(params),method(Method)
    {}

    void processCollisionModel(simulation::Node* node, core::CollisionModel* /*o*/);

    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "collision"; }
    const char* getClassName() const override { return "ColourPickingVisitor"; }

private:

    void processTriangleModel(simulation::Node*, sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>* );
    void processSphereModel(simulation::Node*, sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>*);

    const core::visual::VisualParams* vparams;
    ColourCode method;
};

} // namespace sofa::gui::common
