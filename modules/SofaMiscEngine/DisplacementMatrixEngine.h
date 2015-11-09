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

namespace sofa
{

namespace component
{

namespace engine
{



/*
 * Engine which computes a displacement matrix with respect to an origin matrix, as D(t) = M(t).M(0)^-1
 * @warning Assumes that the initial matrices never change after initialization. This allows computing their inverses only once. To force recomputation, use init().
 * @author François Faure, 2015;
 *
 */
template < class DataTypes >
class DisplacementMatrixEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE( DisplacementMatrixEngine, DataTypes ), sofa::core::DataEngine );

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;
    typedef defaulttype::Mat4x4f Mat4;

    // inputs
    Data< VecCoord > d_x0;  ///< initial bone positions
    Data< VecCoord > d_x;   ///< current bone positions

    // outputs
    Data< helper::vector< Mat4 > > d_displaceMats; ///< displacement matrices


    DisplacementMatrixEngine();
    void init();   // compute the inverse matrices
    void update(); // compute the displacements wrt original positions


    // To simplify the template name in the xml file
    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const DisplacementMatrixEngine<DataTypes>* = NULL) { return DataTypes::Name(); }

protected:
    helper::vector<Mat4> inverses;  ///< inverse initial positions

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_DisplacementMatrixENGINE_H
