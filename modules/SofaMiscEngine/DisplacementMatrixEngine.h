/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_H
#define SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_H

#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{



/*
 * Engine which computes a displacement with respect to an origin position, as D(t) = M(t).M(0)^-1
 * @warning Assumes that the initial positions never change after initialization. This allows computing their inverses only once. To force recomputation, use init().
 * @author François Faure, 2015;
 *
 * Matthieu Nesme, 2015:
 * Generalization of DisplacementMatrixEngine (kept for backward compatibility)
 * The output transform type is a template parameter, it can be a Mat4x4f or a Rigid::Coord (ie translation+quaternion)
 *
 */
template < class DataTypes, class OutputType >
class DisplacementTransformEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE2( DisplacementTransformEngine, DataTypes, OutputType ), sofa::core::DataEngine );

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;

    // inputs
    Data< VecCoord > d_x0;  ///< initial bone positions
    Data< VecCoord > d_x;   ///< current bone positions

    // outputs
    Data< helper::vector< OutputType > > d_displacements; ///< displacement


    DisplacementTransformEngine();
    void init();   // compute the inverse matrices
    void update(); // compute the displacements wrt original positions


    // To simplify the template name in the xml file
    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const DisplacementTransformEngine<DataTypes,OutputType>* = NULL) { return DataTypes::Name()+std::string(",")+defaulttype::DataTypeInfo<OutputType>::name(); }

protected:
    helper::vector<OutputType> inverses;  ///< inverse initial positions

    /// functions that depends on OutputType and must be specialized
    void setInverse( OutputType& inv, const Coord& x0 ); ///< inv = x0^{-1}
    void mult( OutputType& out, const OutputType& inv, const Coord& x ); ///< out = out * inv

};

// Specializations
template <>
void DisplacementTransformEngine<defaulttype::Rigid3Types,defaulttype::Rigid3Types::Coord >::setInverse( defaulttype::Rigid3Types::Coord& inv, const Coord& x0 );
template <>
void DisplacementTransformEngine<defaulttype::Rigid3Types,defaulttype::Rigid3Types::Coord >::mult( defaulttype::Rigid3Types::Coord& out, const defaulttype::Rigid3Types::Coord& inv, const Coord& x );
/////////
template <>
void DisplacementTransformEngine<defaulttype::Rigid3Types,defaulttype::Mat4x4f >::setInverse( defaulttype::Mat4x4f& inv, const Coord& x0 );
template <>
void DisplacementTransformEngine<defaulttype::Rigid3Types,defaulttype::Mat4x4f >::mult( defaulttype::Mat4x4f& out, const defaulttype::Mat4x4f& inv, const Coord& x );



/////////////////////////////////////////////



/*
 * kept for backward compatibility
 */
template < class DataTypes >
class DisplacementMatrixEngine : public DisplacementTransformEngine<DataTypes,defaulttype::Mat4x4f>
{

public:
    SOFA_CLASS( SOFA_TEMPLATE( DisplacementMatrixEngine, DataTypes ),SOFA_TEMPLATE2( DisplacementTransformEngine, DataTypes, defaulttype::Mat4x4f ) );


    DisplacementMatrixEngine()
        : DisplacementTransformEngine<DataTypes,defaulttype::Mat4x4f>()
    {
        this->d_displacements.setName( "displaceMats" );
        this->addAlias( &this->d_displacements, "displaceMats" );
    }

    // To simplify the template name in the xml file
    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const DisplacementMatrixEngine<DataTypes>* = NULL) { return DataTypes::Name(); }
};





} // namespace engine

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_DisplacementMatrixENGINE_H
