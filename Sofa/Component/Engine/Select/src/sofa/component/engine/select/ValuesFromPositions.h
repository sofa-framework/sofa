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
#include <sofa/component/engine/select/config.h>



#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::engine::select
{

/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given box.
 */
template <class DataTypes>
class ValuesFromPositions : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ValuesFromPositions,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef type::vector<Real> VecReal;
    typedef typename DataTypes::CPos CPos;
    typedef type::Vec<3, Real> Vec3;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;

protected:

    ValuesFromPositions();

    ~ValuesFromPositions() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            {
                arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                              "' found in the context node.");
                return false; // this template is not the same as the existing MechanicalState
            }
        }

        return BaseObject::canCreate(obj, context, arg);
    }

protected:
    struct TempData
    {
        CPos dir;
        Real bmin, bmax;
        VecReal inputValues;
        const VecCoord* x0;
    };

    void updateValues(TempData& _data);
    Real valueFromPosition(const CPos& p, const TempData& data);
    Real valueFromPoint(const PointID& pid, const TempData& data);
    Real valueFromEdge(const Edge& e, const TempData& data);
    Real valueFromTriangle(const Triangle& t, const TempData& data);
    Real valueFromTetrahedron(const Tetra &t, const TempData& data);

    void updateVectors(TempData& _data);
    Vec3 vectorFromPosition(const CPos& p, const TempData& data);
    Vec3 vectorFromPoint(const PointID& pid, const TempData& data);
    Vec3 vectorFromEdge(const Edge& e, const TempData& data);
    Vec3 vectorFromTriangle(const Triangle& t, const TempData& data);
    Vec3 vectorFromTetrahedron(const Tetra &t, const TempData& data);

public:
    //Input
    Data<VecReal> f_inputValues; ///< Input values
    Data<CPos> f_direction; ///< Direction along which the values are interpolated
    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    Data<type::vector<Edge> > f_edges; ///< Edge Topology
    Data<type::vector<Triangle> > f_triangles; ///< Triangle Topology
    Data<type::vector<Tetra> > f_tetrahedra; ///< Tetrahedron Topology

    //Output scalars
    Data<VecReal> f_values; ///< Values of the points contained in the ROI
    Data<VecReal> f_edgeValues; ///< Values of the edges contained in the ROI
    Data<VecReal> f_triangleValues; ///< Values of the triangles contained in the ROI
    Data<VecReal> f_tetrahedronValues; ///< Values of the tetrahedra contained in the ROI

    //Output vectors
    Data<sofa::type::vector<Vec3> > f_pointVectors; ///< Vectors of the points contained in the ROI
    Data<sofa::type::vector<Vec3> > f_edgeVectors; ///< Vectors of the edges contained in the ROI
    Data<sofa::type::vector<Vec3> > f_triangleVectors; ///< Vectors of the triangles contained in the ROI
    Data<sofa::type::vector<Vec3> > f_tetrahedronVectors; ///< Vectors of the tetrahedra contained in the ROI

    // parameters
    sofa::core::objectmodel::Data< sofa::helper::OptionsGroup > p_fieldType; ///< field type of output elements
    Data <bool> p_drawVectors; ///< draw vectors line
    Data <float> p_vectorLength; ///< vector length visualisation. 
};

#if !defined(SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromPositions<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromPositions<defaulttype::Rigid3Types>; 
#endif

} //namespace sofa::component::engine::select
