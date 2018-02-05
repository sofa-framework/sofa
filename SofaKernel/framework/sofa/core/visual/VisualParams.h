/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/system/gl.h>


namespace sofa
{

namespace helper
{
namespace gl
{
class FrameBufferObject;
} // namespace gl
} // namespace helper

namespace core
{
namespace visual
{

class DrawTool;

/// The enumeration used to describe potentially supported graphics API.
enum
{
    API_OpenGL = 0,
    API_OGRE = 1,
    API_OpenSceneGraph = 2,
    API_OpenSG = 3
};

/// Class gathering parameters used by visual components and by the draw method of each component. Transmitted by visitors
class SOFA_CORE_API VisualParams : public ExecParams
{
public:
	typedef sofa::helper::fixed_array<int, 4> Viewport;

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

    VisualParams();

    /// Get the default VisualParams, to be used to provide a default values for method parameters
    static VisualParams* defaultInstance();

    const Viewport& viewport() const { return m_viewport; }
    Viewport& viewport() { return m_viewport; }

    const SReal& zNear() const { return m_zNear; }
    const SReal& zFar()  const { return m_zFar;  }
    SReal& zNear() { return m_zNear; }
    SReal& zFar()  { return m_zFar;  }

    const CameraType& cameraType() const { return m_cameraType; }
    CameraType& cameraType() { return m_cameraType; }

    const Pass& pass() const { return m_pass; }
    Pass& pass() { return m_pass; }

    DrawTool*& drawTool() { return m_drawTool; }
    DrawTool*& drawTool() const { return m_drawTool; }

    helper::gl::FrameBufferObject*& frameBufferObject() { return m_boundFrameBuffer; }
    helper::gl::FrameBufferObject*& frameBufferObject() const { return m_boundFrameBuffer; }

    DisplayFlags& displayFlags() { return m_displayFlags; }
    const DisplayFlags& displayFlags() const { return m_displayFlags; }

    sofa::defaulttype::BoundingBox&  sceneBBox()    { return m_sceneBoundingBox; }
    const sofa::defaulttype::BoundingBox&  sceneBBox() const   { return m_sceneBoundingBox; }

    /// Store the ModelView matrix used to draw the scene. This OpenGL matrix defines the world coordinate system with respect to the camera.
    void setModelViewMatrix( const double m[16] ) { for(unsigned i=0; i<16; i++) m_modelViewMatrix[i] = m[i]; }

    /// Get the ModelView matrix used to draw the scene. This OpenGL matrix defines the world coordinate system with respect to the camera.
    void getModelViewMatrix( double m[16] ) const { for(unsigned i=0; i<16; i++) m[i] = m_modelViewMatrix[i]; }

    /// Store the projection matrix used to draw the scene. This OpenGL matrix defines the camera coordinate system with respect to the viewport, including perspective if any.
    void setProjectionMatrix( const double m[16] ) { for(unsigned i=0; i<16; i++) m_projectionMatrix[i] = m[i]; }

    /// Get the projection matrix used to draw the scene. This OpenGL matrix defines the camera coordinate system with respect to the viewport, including perspective if any.
    void getProjectionMatrix( double m[16] ) const { for(unsigned i=0; i<16; i++) m[i] = m_projectionMatrix[i]; }

    /// @todo clarify what this is with respect to ModelView and Perspective matrices
    sofa::helper::gl::Transformation& sceneTransform() { return m_sceneTransform; }
    const sofa::helper::gl::Transformation& sceneTransform() const { return m_sceneTransform; }


    bool isSupported(unsigned int api) const
    {
        return (m_supportedAPIs & (1<<api)) != 0;
    }

    void setSupported(unsigned int api, bool val=true)
    {
        if (val)
            m_supportedAPIs |= (1<<api);
        else
            m_supportedAPIs &= ~(1<<api);
    }

protected:
    sofa::defaulttype::BoundingBox      m_sceneBoundingBox;
    helper::gl::Transformation          m_sceneTransform;
    Viewport                            m_viewport;
    SReal                              m_zNear;
    SReal                              m_zFar;
    CameraType                          m_cameraType;
    Pass                                m_pass;
    DisplayFlags                        m_displayFlags;
    mutable DrawTool*                   m_drawTool;
    mutable helper::gl::FrameBufferObject*	m_boundFrameBuffer;
    /// Ids of position vector
    ConstMultiVecCoordId m_x;
    /// Ids of velocity vector
    ConstMultiVecDerivId m_v;
    /// Mask of supported graphics API
    unsigned int m_supportedAPIs;

    SReal m_modelViewMatrix[16];  ///< model view matrix.
    SReal m_projectionMatrix[16]; ///< projection matrix.
};

} // namespace visual
} // namespace core
} // namespace sofa

#endif // SOFA_CORE_VISUAL_VISUALPARAMS_H
