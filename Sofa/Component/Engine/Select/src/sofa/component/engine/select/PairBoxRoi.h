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
#include <sofa/core/VecId.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::select
{

/**
 * This class find all the points located between two boxes. The difference between the inclusive box (surrounding the mesh) and the included box (inside the mesh) gives the border points of the mesh.
 */
template <class DataTypes>
class PairBoxROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PairBoxROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef type::Vec<6,Real> Vec6;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
 
protected:

    PairBoxROI();

    ~PairBoxROI() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams*) override;

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
    bool isPointInBox(const CPos& p, const Vec6& b);
    bool isPointInBox(const PointID& pid, const Vec6& b);

public:
    //Input
    /// A box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    //Box surrounding the mesh
    Data<Vec6> inclusiveBox; ///< Inclusive box defined by xmin,ymin,zmin, xmax,ymax,zmax
    //Box inside the mesh 
    Data<Vec6> includedBox; ///< Included box defined by xmin,ymin,zmin, xmax,ymax,zmax
    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    // Point coordinates of the mesh in 3D in double.
    Data <VecCoord> positions; ///< Vertices of the mesh loaded
   

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI

    //Parameter
    Data<bool> p_drawInclusiveBox; ///< Draw Inclusive Box
    Data<bool> p_drawIncludedBox; ///< Draw Included Box
    Data<bool> p_drawPoints; ///< Draw Points
    Data<double> _drawSize; ///< Draw Size
};

#if !defined(SOFA_COMPONENT_ENGINE_PAIRBOXROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<defaulttype::Vec6Types>; //Phuoc
 
#endif

} //namespace sofa::component::engine::select
