/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec3Types.h>

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

    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

private:
    core::behavior::MechanicalState<DataTypes>* nodes;
    core::topology::BaseMeshTopology* topo;

    Data<bool> depthTest;
    Data<bool> blending;

public:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();

    void init();
    void drawTransparent(const core::visual::VisualParams* );
    bool addBBox(double* minBBox, double* maxBBox);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const OglTetrahedralModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

};

}
}
}

#endif /*OGLTETRAHEDRALMODEL_H_*/
