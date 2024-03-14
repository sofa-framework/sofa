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
#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Quat.h>

namespace sofa::core::visual
{

class VisualParams;

/**
 *  \brief An interface which all VisualModel inherit.
 *
 *  This Interface is used for the VisualModel, which all visible objects must
 *  implement.
 *
 *  VisualModels are drawn by calling their draw method. The method update is
 *  used to recompute some internal data (such as normals) after the simulation
 *  has computed a new timestep.
 *
 *  Most VisualModel are bound by a Mapping to a BehaviorModel or
 *  MechanicalState.
 */
class SOFA_CORE_API VisualModel : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(VisualModel, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(VisualModel)

    Data<bool> d_enable; ///< Display the visual model or not

    /**
     *  \brief Display the VisualModel object.
     *
     *  TODO(dmarchal, 2023-06-09): Deprecate VI and use NVI design pattern: In one year, remove the virtual keyword so that everyone
     *  will have to override doDrawVisual;
     */
    virtual void drawVisual(const VisualParams* /*vparams*/) final;


protected:
    VisualModel();
    ~VisualModel() override { }

private:
    virtual void doDrawVisual(const VisualParams* /*vparams*/) {}

public:
    /**
     *  \brief Initialize the textures, or other graphical resources.
     *
     *  Called once before the first frame is drawn, and if the graphical
     *  context has been recreated.
     */
    virtual void initVisual() {  }

    /**
     *  \brief clear some graphical resources (generaly called before the deleteVisitor).
     *  \note: for more general usage you can use the cleanup visitor
     */
    virtual void clearVisual() { }

    /**
     *  \brief Called before objects in the current branch are displayed
     */
    virtual void fwdDraw(VisualParams* /*vparams*/) {}

    /**
     *  \brief Called after objects in the current branch are displayed
     */
    virtual void bwdDraw(VisualParams* /*vparams*/) {}

    /**
     *  \brief Display transparent surfaces.
     *
     *  Transparent objects should use this method to get a correct display order.
     */
    virtual void drawTransparent(const VisualParams* /*vparams*/)
    {
    }

    /**
     *  \brief Display shadow-casting surfaces.
     *
     *  This method default to calling draw(). Object that do not cast any
     *  shadows, or that use a different LOD for them should reimplement it.
     */
    virtual void drawShadow(const VisualParams* vparams)
    {
        doDrawVisual(vparams);
    }

    /**
     *  \brief used to update the model if necessary.
     */
    virtual void updateVisual() {  }
    /**
    *  \brief used to update the model if necessary.
    */
    virtual void parallelUpdateVisual() { }


    /**
     *  \brief used to add the bounding-box of this visual model to the
     *  given bounding box in order to compute the scene bounding box or
     *  cull hidden objects.
     *
     *  \return false if the visual model does not define any bounding box,
     *  which should only be the case for "debug" objects, as this lack of
     *  information might affect performances and leads to incorrect scene
     *  bounding box.
     */
    virtual bool addBBox(SReal* /*minBBox*/, SReal* /*maxBBox*/)
    {
        return false;
    }

    /// Translate the positions
    ///
    /// This method is optional, it is used when the user want to interactively change the position of an object
    virtual void applyTranslation(const SReal /*dx*/, const SReal /*dy*/, const SReal /*dz*/)
    {
    }

    /// Rotate the positions using Euler Angles in degree
    ///
    /// This method is optional, it is used when the user want to interactively change the position of an object
    virtual void applyRotation (const SReal /*rx*/, const SReal /*ry*/, const SReal /*rz*/)
    {
    }

    /// Rotate the positions
    ///
    /// This method is optional, it is used when the user want to interactively change the position of an object
    virtual void applyRotation(const type::Quat<SReal> /*q*/)
    {
    }

    /// Scale the positions
    ///
    /// This method is optional, it is used when the user want to interactively change the position of an object
    virtual void applyScale(const SReal /*sx*/,const SReal /*sy*/,const SReal /*sz*/)
    {
    }

    /**
     *  \brief Append this mesh to an OBJ format stream.
     *
     *  The number of vertices position, normal, and texture coordinates already written is given as parameters.
     *  This method should update them.
     */
    virtual void exportOBJ(std::string /*name*/, std::ostream* /*out*/, std::ostream* /*mtl*/, 
        sofa::Index& /*vindex*/, sofa::Index& /*nindex*/, sofa::Index& /*tindex*/, int& /*count*/)
    {
    }

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;
};
} // namespace sofa::core::visual

