/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGLTETRAHEDRALMODEL_H_
#define OGLTETRAHEDRALMODEL_H_
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{
namespace component
{
namespace visualmodel
{

/**
 *  \brief Render 3D models with tetrahedra.
 *
 *  This is a basic class using tetrehedra for the rendering
 *  instead of common triangles. It loads its data with
 *  a BaseMeshTopology and a MechanicalState.
 *  This rendering is only available with Nvidia's >8 series
 *  and Ati's >2K series.
 *
 */

template<class DataTypes>
class OglTetrahedralModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglTetrahedralModel, core::visual::VisualModel);
    //typedef ExtVec3fTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    core::topology::BaseMeshTopology* m_topology;

    topology::PointData< sofa::defaulttype::ResizableExtVector<Coord> > m_positions;
    Data< sofa::defaulttype::ResizableExtVector<Tetrahedron> > m_tetrahedrons;
    bool modified;
    int lastMeshRev;
    bool useTopology;



private:
    

    Data<bool> depthTest;
    Data<bool> blending;

protected:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();
public:
    void init();
    void drawTransparent(const core::visual::VisualParams* vparams);
    void computeBBox(const core::ExecParams *, bool onlyVisible=false);

    virtual void updateVisual();
    virtual void computeMesh();
    //virtual std::string getTemplateName() const
    //{
    //    return templateName(this);
    //}

    static std::string templateName(const OglTetrahedralModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    //template<class T>
    //static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    //{
    //    if (context->getMechanicalState() == NULL)
    //        return false;
    //    return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    //}

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_VISUALMODEL_OGLTETRAHEDRALMODEL_CPP)
#ifndef SOFA_FLOAT
extern template class OglTetrahedralModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class OglTetrahedralModel<defaulttype::Vec3fTypes>;
#endif
#endif

}
}
}

#endif /*OGLTETRAHEDRALMODEL_H_*/
