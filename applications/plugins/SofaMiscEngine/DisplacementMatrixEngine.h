/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_ENGINE_DISPLACEMENTMATRIXENGINE_H
#define SOFA_COMPONENT_ENGINE_DISPLACEMENTMATRIXENGINE_H

#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaMiscEngine/DisplacementTransformEngine.h>

namespace sofa
{
namespace component
{
namespace engine
{

///
/// kept for backward compatibility now deprecated by DisplacementTransformEngine
///
template < class DataTypes >
class SOFA_MISC_ENGINE_API DisplacementMatrixEngine : public DisplacementTransformEngine<DataTypes, defaulttype::Mat4x4f>
{

public:
    SOFA_CLASS( SOFA_TEMPLATE( DisplacementMatrixEngine, DataTypes ),
                SOFA_TEMPLATE2( DisplacementTransformEngine, DataTypes, defaulttype::Mat4x4f ) );

    typedef DisplacementTransformEngine<DataTypes, defaulttype::Mat4x4f> Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename defaulttype::Mat4x4f Matrix4x4;
    typedef typename sofa::defaulttype::Vec<4,float> Line;

    /// Method
    DisplacementMatrixEngine();

    void init();   /// compute the inverse matrices
    void reinit(); /// compute S*inverse and store it once and for all.
    void update(); /// compute the displacements wrt original positions

    /// To simplify the template name in the xml file
    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const DisplacementMatrixEngine<DataTypes>* = NULL) { return DataTypes::Name(); }

    /// inputs
    Data< helper::vector< sofa::defaulttype::Vec<3,Real> > > d_scales; ///< scale matrices
    helper::vector<Matrix4x4> SxInverses;  ///< inverse initial positions
};

} /// namespace engine
} /// namespace component
} /// namespace sofa

#endif /// SOFA_COMPONENT_ENGINE_DISPLACEMENTMATRIXENGINE_H
