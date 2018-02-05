/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_COLOURPICKING_VISITOR
#define SOFA_GUI_COLOURPICKING_VISITOR

#include <sofa/gui/SofaGUI.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/ExecParams.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaUserInteraction/MouseInteractor.h>

namespace sofa
{

namespace gui
{

void decodeCollisionElement( const sofa::defaulttype::Vec4f colour, sofa::component::collision::BodyPicked& body );
void decodePosition( sofa::component::collision::BodyPicked& body, const sofa::defaulttype::Vec4f colour, const component::collision::TriangleModel* model,
        const unsigned int index);
void decodePosition( sofa::component::collision::BodyPicked& body, const sofa::defaulttype::Vec4f colour, const component::collision::SphereModel* model,
        const unsigned int index);


/* Launches the drawColourPicking() method of each CollisionModel */
class SOFA_SOFAGUI_API ColourPickingVisitor : public simulation::Visitor
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
        :simulation::Visitor(params),vparams(params),method(Method)
    {}

    void processCollisionModel(simulation::Node* node, core::CollisionModel* /*o*/);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "collision"; }
    virtual const char* getClassName() const { return "ColourPickingVisitor"; }

private:

    void processTriangleModel(simulation::Node*, sofa::component::collision::TriangleModel* );
    void processSphereModel(simulation::Node*, sofa::component::collision::SphereModel*);

    const core::visual::VisualParams* vparams;
    ColourCode method;
};



}
}



#endif // SOFA_GUI_COLOURPICKING_VISITOR
