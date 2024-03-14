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

#include <sofa/component/io/mesh/config.h>

#include <sofa/core/State.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/Visitor.h>

#if SOFA_COMPONENT_IO_MESH_HAVE_ZLIB
#include <zlib.h>
#endif

#include <fstream>

namespace sofa::component
{
namespace _blenderexporter_
{

/**
* This component can be used to export the physical simulation result to Blender (http://www.blender.org/) 
* by replacing an existing cached simulation.
* This exporter create a sequence of .bphys file containing the simulation state at each frame.
* These files must be copied (or directly writed) in a blendcache folder created in the same directory 
* that the blender file.
*/

// TODO: currently the export only support soft body and hair simulations, clothes, smoke and fluid simulation could be added.

template<class T>
class SOFA_COMPONENT_IO_MESH_API BlenderExporter: public core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherit;
    typedef sofa::core::State<T> DataType;
    typedef typename DataType::VecCoord VecCoord;
    typedef typename DataType::VecDeriv VecDeriv;
    typedef typename DataType::Coord Coord;
    typedef typename DataType::Deriv Deriv;
    typedef typename DataType::ReadVecDeriv ReadVecDeriv;
    typedef typename DataType::ReadVecCoord ReadVecCoord;

    typedef enum{SoftBody,Particle,Cloth,Hair}SimulationType;

    SOFA_CLASS(SOFA_TEMPLATE(BlenderExporter,T),core::objectmodel::BaseObject);

    Data < std::string > path; ///< output path
    Data < std::string > baseName; ///< Base name for the output files
    Data < int > simulationType; ///< simulation type (0: soft body, 1: particles, 2:cloth, 3:hair)
    Data < int > simulationStep; ///< save the  simulation result every step frames
    Data < int > nbPtsByHair; ///< number of element by hair strand

protected:

    typename DataType::SPtr mmodel;

    BlenderExporter();

    ~BlenderExporter() override{}

public:

    static const char* Name(){return "Blender exporter";}

    void init() override;

    void reset() override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T2>
    static bool canCreate(T2*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<DataType*>(context->getState()) == nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + T::Name() +
                          "' found in the context node.");
            return false;
        }
        return BaseObject::canCreate(obj, context, arg);
    }

protected:

    unsigned frameCounter;
};

} // namespace _blenderexporter_

namespace exporter {
    template<class T>
    using BlenderExporter = _blenderexporter_::BlenderExporter<T>;
} // namespace exporter

} // namespace sofa::component
