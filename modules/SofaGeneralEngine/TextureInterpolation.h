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
#ifndef SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_H
#define SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef sofa::defaulttype::Vec<1,Real>                       Coord1D;
    typedef sofa::defaulttype::Vec<2,Real>                       Coord2D;
    typedef sofa::defaulttype::Vec<3,Real>                       Coord3D;
    typedef sofa::defaulttype::ResizableExtVector <Coord2D>      ResizableExtVector2D;
    typedef sofa::helper::vector <Coord3D>    VecCoord3D;


protected:

    TextureInterpolation();

    ~TextureInterpolation() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    void draw(const core::visual::VisualParams* vparams) override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const TextureInterpolation<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:

    /// input state vector
    Data <VecCoord> _inputField;

    /// input coordinate vector (optional)
    Data <VecCoord3D> _inputCoords;

    /// output texture coordinate vector
    Data <ResizableExtVector2D> _outputCoord;

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
    Data <std::map < std::string, sofa::helper::vector<Real> > > f_graph; ///< Vertex state value per iteration

    void updateGraph();
    void resetGraph();

    void standardLinearInterpolation();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec1dTypes>;
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec2dTypes>;
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec1fTypes>;
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec2fTypes>;
extern template class SOFA_GENERAL_ENGINE_API TextureInterpolation<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
