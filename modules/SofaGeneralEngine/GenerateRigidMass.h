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
#ifndef SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_H
#define SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes, class MassType>
class GenerateRigidMass : public core::DataEngine
{
public:
    SOFA_CLASS(GenerateRigidMass,core::DataEngine);

    GenerateRigidMass();
    ~GenerateRigidMass();

    /// Initialization method called at graph modification, during bottom-up traversal.
    virtual void init() override;
    /// Update method called when variables used in precomputation are modified.
    virtual void reinit() override;
    /// Update the output values
    virtual void update() override;

protected:

    typedef defaulttype::Vector3 Vector3;
    typedef helper::fixed_array <unsigned int,3> MTriangle;
    typedef helper::fixed_array <unsigned int,4> MQuad;
    typedef helper::vector <unsigned int> MPolygon;

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Vec3 Vec3;
    typedef defaulttype::Mat<3,3,Real> Mat3x3;

    /**
      * Data Fields
      */
    /// input
    Data< Real > m_density; // kg * m^-3
    Data< helper::vector< Vector3 > > m_positions;
    Data< helper::vector< MTriangle > > m_triangles;
    Data< helper::vector< MQuad > > m_quads;
    Data< helper::vector< MPolygon > > m_polygons; // must be convex

    /// output
    Data< MassType > rigidMass;
    Data< Real > mass;
    Data< Real > volume;
    Data < Mat3x3 > inertiaMatrix;
    Data< Vec3 > massCenter;
    Data< Vector3 > centerToOrigin;

    /**
      * Protected methods
      */
    /// integrates the whole mesh
    void integrateMesh();

    void integrateTriangle(Vector3 kV0,Vector3 kV1,Vector3 kV2);

    /// generates the RigidMass from the mesh integral
    void generateRigid();

    helper::fixed_array<SReal,10> afIntegral;

public:

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return core::DataEngine::canCreate (obj, context, arg);
    }

    virtual std::string getTemplateName() const override;
    static std::string templateName(const GenerateRigidMass<DataTypes,MassType>*);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API GenerateRigidMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API GenerateRigidMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>;
#endif
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
