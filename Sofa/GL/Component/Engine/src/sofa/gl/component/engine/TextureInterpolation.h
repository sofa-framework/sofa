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
#include <sofa/gl/component/engine/config.h>



#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/map.h>

namespace sofa::gl::component::engine
{

/**
 * This class give texture coordinate in 1D according to an imput state vector.
 */
template <class DataTypes>
class TextureInterpolation : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TextureInterpolation,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord         Coord;
    typedef typename DataTypes::VecCoord      VecCoord;
    typedef typename DataTypes::Real          Real;
    typedef sofa::type::Vec<1,Real>                       Coord1D;
    typedef sofa::type::Vec<2,Real>                       Coord2D;
    typedef sofa::type::Vec<3,Real>                       Coord3D;
    typedef type::vector<Coord2D>          VecCoord2D;
    typedef sofa::type::vector<Coord3D>    VecCoord3D;


protected:

    TextureInterpolation();

    ~TextureInterpolation() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    /// input state vector
    Data <VecCoord> _inputField;

    /// input coordinate vector (optional)
    Data <VecCoord3D> _inputCoords;

    /// output texture coordinate vector
    Data <VecCoord2D> _outputCoord;

    /// bool used to specify scalar input field (if higher template is needed)
    Data<bool> _scalarField;

    /// Data for interpolation scale:
    Data <Real> _minVal;
    Data <Real> _maxVal; ///< maximum value of state value for interpolation.
    Data <bool> _changeScale; ///< compute texture interpolation on manually scale defined above.

    /// Data for interpolation scale:
    Data <bool> drawPotentiels;
    Data <float> showIndicesScale; ///< Debug : scale of state values displayed.

    Data <unsigned int> _vertexPloted; ///< Vertex index of values display in graph for each iteration.
    Data <std::map < std::string, sofa::type::vector<Real> > > f_graph; ///< Vertex state value per iteration

    void updateGraph();
    void resetGraph();

    void standardLinearInterpolation();
};

#if !defined(SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_CPP)
extern template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation<defaulttype::Vec1Types>;
extern template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation<defaulttype::Vec2Types>;
extern template class SOFA_GL_COMPONENT_ENGINE_API TextureInterpolation<defaulttype::Vec3Types>;
#endif

} //namespace sofa::gl::component::engine
