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
#ifndef SOFA_COMPONENT_ENGINE_GENERATECYLINDER_H
#define SOFA_COMPONENT_ENGINE_GENERATECYLINDER_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/SofaGeneral.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class dilates the positions of one DataFields into new positions after applying a dilateation
This dilateation can be either translation, rotation, scale
 */
template <class DataTypes>
class GenerateCylinder : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GenerateCylinder,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef typename SeqTetrahedra::value_type Tetrahedron;

public:

    GenerateCylinder();

    ~GenerateCylinder() {}

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const GenerateCylinder<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    Data<VecCoord> f_outputX; ///< ouput position
    Data<SeqTetrahedra> f_tetrahedron; ///< output tetrahedra
    Data<Real > f_radius; /// radius of cylinder 
	Data<Real > f_height; /// height of cylinder
    Data<Coord> f_origin; /// origin
    Data<size_t> f_resolutionCircumferential; /// number of points in the circumferential direction
    Data<size_t> f_resolutionRadial; /// number of points in the radial  direction
    Data<size_t> f_resolutionHeight; /// number of points in the height direction
};



} // namespace engine

} // namespace component

} // namespace sofa

#endif
