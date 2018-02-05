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
#ifndef SOFA_COMPONENT_ENGINE_PAIRBOXROI_H
#define SOFA_COMPONENT_ENGINE_PAIRBOXROI_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/VecId.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
 
protected:

    PairBoxROI();

    ~PairBoxROI() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    void draw(const core::visual::VisualParams*) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false; // this template is not the same as the existing MechanicalState
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::create(tObj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const PairBoxROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected:
    bool isPointInBox(const CPos& p, const Vec6& b);
    bool isPointInBox(const PointID& pid, const Vec6& b);

public:
    //Input
    /// A box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    //Box surrounding the mesh
    Data<Vec6> inclusiveBox; 
    //Box inside the mesh 
    Data<Vec6> includedBox; 
    Data<VecCoord> f_X0;
    // Point coordinates of the mesh in 3D in double.
    Data <VecCoord> positions; 
   

    //Output
    Data<SetIndex> f_indices;
    Data<VecCoord > f_pointsInROI;

    //Parameter
    Data<bool> p_drawInclusiveBox;
    Data<bool> p_drawIncludedBox;
    Data<bool> p_drawPoints;
    Data<double> _drawSize;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_PAIRBOXROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Rigid3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Vec6dTypes>; //Phuoc
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Rigid3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API PairBoxROI<defaulttype::Vec6fTypes>; //Phuoc
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
