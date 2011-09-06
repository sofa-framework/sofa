/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_CORE_VISUAL_VISUALPARAMS_H
#define SOFA_CORE_VISUAL_VISUALPARAMS_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/defaulttype/Vec.h>


namespace sofa
{
namespace core
{
namespace visual
{

/// Class gathering parameters used by visual components and by the draw method of each component. Transmitted by visitors
class VisualParams : public ExecParams
{
public:

    typedef sofa::helper::fixed_array<GLint,4> Viewport;

    /// The enumeration used to describe each step of the rendering.
    enum Pass
    {
        Std,		///< Standard pass
        Transparent,	///< Transparent pass
        Shadow 		///< Shadow pass
    };

    /// The enumeration used to describe the type of camera transform.
    enum CameraType
    {
        PERSPECTIVE_TYPE =0, ///< Perspective camera
        ORTHOGRAPHIC_TYPE =1 ///< Orthographic camera
    };

    /// @name Access to vectors from a given state container (i.e. State or MechanicalState)
    /// @{

    /// Read access to current position vector
    template<class S>
    const Data<typename S::VecCoord>* readX(const S* state) const
    {   return m_x[state].read();    }

    /// Read access to current velocity vector
    template<class S>
    const Data<typename S::VecDeriv>* readV(const S* state) const
    {   return m_v[state].read();    }

    /// @}

    /// @name Setup methods
    /// Called by the OdeSolver from which the mechanical computations originate.
    /// They all return a reference to this MechanicalParam instance, to ease chaining multiple setup calls.
    /// @{
    const ConstMultiVecCoordId& x() const { return m_x; }
    ConstMultiVecCoordId& x()       { return m_x; }

    const ConstMultiVecDerivId& v() const { return m_v; }
    ConstMultiVecDerivId& v()       { return m_v; }

    /// Set the IDs of position vector
    VisualParams& setX(                   ConstVecCoordId v) { m_x.assign(v);   return *this; }
    VisualParams& setX(                   ConstMultiVecCoordId v) { m_x = v;    return *this; }
    template<class StateSet>
    VisualParams& setX(const StateSet& g, ConstVecCoordId v) { m_x.setId(g, v); return *this; }

    /// Set the IDs of velocity vector
    VisualParams& setV(                   ConstVecDerivId v) { m_v.assign(v);   return *this; }
    VisualParams& setV(                   ConstMultiVecDerivId v) { m_v = v;    return *this; }
    template<class StateSet>
    VisualParams& setV(const StateSet& g, ConstVecDerivId v) { m_v.setId(g, v); return *this; }
    /// @}

    VisualParams()
        :m_viewport(sofa::helper::make_array(0,0,0,0))
        ,m_zNear(0)
        ,m_zFar(0)
        ,m_cameraType(PERSPECTIVE_TYPE)
        ,m_pass(Std)
        ,m_drawTool(NULL)
        ,m_x (ConstVecCoordId::position())
        ,m_v (ConstVecDerivId::velocity())
    {
    }

    static VisualParams* defaultInstance()
    {
        static VisualParams m_defaultInstance;
        return &m_defaultInstance;
    }


    const Viewport& viewport() const { return m_viewport; }
    Viewport& viewport() { return m_viewport; }

    const double& zNear() const { return m_zNear; }
    const double& zFar()  const { return m_zFar;  }
    double& zNear() { return m_zNear; }
    double& zFar()  { return m_zFar;  }

    const CameraType& cameraType() const { return m_cameraType; }
    CameraType& cameraType() { return m_cameraType; }

    const Pass& pass() const { return m_pass; }
    Pass& pass() { return m_pass; }

    DrawTool*& drawTool() { return m_drawTool; }
    DrawTool*& drawTool() const { return m_drawTool; }

    DisplayFlags& displayFlags() { return m_displayflags; }
    const DisplayFlags& displayFlags() const { return m_displayflags; }

    sofa::defaulttype::BoundingBox&  sceneBBox()    { return m_sceneBoundingBox; }
    const sofa::defaulttype::BoundingBox&  sceneBBox() const   { return m_sceneBoundingBox; }

    sofa::helper::gl::Transformation& sceneTransform() { return m_sceneTransform; }
    const sofa::helper::gl::Transformation& sceneTransform() const { return m_sceneTransform; }

protected:
    sofa::defaulttype::BoundingBox      m_sceneBoundingBox;
    helper::gl::Transformation          m_sceneTransform;
    Viewport                            m_viewport;
    double                              m_zNear;
    double                              m_zFar;
    CameraType                          m_cameraType;
    Pass                                m_pass;
    DisplayFlags                        m_displayflags;
    mutable DrawTool*                   m_drawTool;
    /// Ids of position vector
    ConstMultiVecCoordId m_x;
    /// Ids of velocity vector
    ConstMultiVecDerivId m_v;

};



}//visual
}//core
}//sofa

#endif // SOFA_CORE_VISUAL_VISUALPARAMS_H
