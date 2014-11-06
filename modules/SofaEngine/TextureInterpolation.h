/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_H
#define SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/SofaGeneral.h>
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
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

    virtual std::string getTemplateName() const
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
    Data <Real> _maxVal;
    Data <bool> _changeScale;

    /// Data for interpolation scale:
    Data <bool> drawPotentiels;
    Data <float> showIndicesScale;

    Data <unsigned int> _vertexPloted;
    Data <std::map < std::string, sofa::helper::vector<Real> > > f_graph;

    void updateGraph();
    void resetGraph();

    void standardLinearInterpolation();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API TextureInterpolation<defaulttype::Vec1dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API TextureInterpolation<defaulttype::Vec1fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
