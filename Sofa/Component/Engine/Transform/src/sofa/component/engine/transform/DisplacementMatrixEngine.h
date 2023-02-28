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

#include <sofa/component/engine/transform/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::transform
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
 * Ali Dicko, 2015
 * Add of a data scale to the DisplacementMatrixEngine which is a scale matrix add to the computation of D(t). Now D(t) = M(t).M(0)^-1.S
 * By default S is the identity matrix, but the user can set other matrices
 */
template < class DataTypes, class OutputType >
class DisplacementTransformEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE2( DisplacementTransformEngine, DataTypes, OutputType ), sofa::core::DataEngine );

    typedef sofa::core::DataEngine Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;

    // inputs
    Data< VecCoord > d_x0;  ///< initial bone positions
    Data< VecCoord > d_x;   ///< current bone positions

    // outputs
    Data< type::vector< OutputType > > d_displacements; ///< displacement

    // methods
    DisplacementTransformEngine();
    void init() override;   // compute the inverse matrices
    void doUpdate() override; // compute the displacements wrt original positions

protected:
    type::vector<OutputType> inverses;  ///< inverse initial positions

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
void DisplacementTransformEngine<defaulttype::Rigid3Types, type::Mat4x4 >::setInverse( type::Mat4x4& inv, const Coord& x0 );
template <>
void DisplacementTransformEngine<defaulttype::Rigid3Types,type::Mat4x4 >::mult( type::Mat4x4& out, const type::Mat4x4& inv, const Coord& x );

/////////////////////////////////////////////

/*
 * kept for backward compatibility
 */
template < class DataTypes >
class DisplacementMatrixEngine : public DisplacementTransformEngine<DataTypes, type::Mat4x4>
{

public:
    SOFA_CLASS( SOFA_TEMPLATE( DisplacementMatrixEngine, DataTypes ),SOFA_TEMPLATE2( DisplacementTransformEngine, DataTypes, type::Mat4x4 ) );

    typedef DisplacementTransformEngine<DataTypes, type::Mat4x4> Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename sofa::type::Mat<4,4,Real> Matrix4x4;
    typedef typename sofa::type::Vec<4,Real> Line;

    // Method
    DisplacementMatrixEngine();

    void init() override;   // compute the inverse matrices
    void reinit() override; // compute S*inverse and store it once and for all.
    void doUpdate() override; // compute the displacements wrt original positions

    // inputs
    Data< type::vector< sofa::type::Vec<3,Real> > > d_scales; ///< scale matrices
    type::vector<Matrix4x4> SxInverses;  ///< inverse initial positions
};

} // namespace sofa::component::engine::transform
