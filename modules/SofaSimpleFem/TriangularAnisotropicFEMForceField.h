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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "TriangularFEMForceField.h"
#include <SofaBaseTopology/TopologyData.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>



namespace sofa
{
namespace component
{
namespace forcefield
{

using sofa::helper::vector;


template<class DataTypes>
class TriangularAnisotropicFEMForceField : public sofa::component::forcefield::TriangularFEMForceField<DataTypes>
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularAnisotropicFEMForceField, DataTypes), SOFA_TEMPLATE(TriangularFEMForceField, DataTypes));

    typedef sofa::component::forcefield::TriangularFEMForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename Inherited::TriangleInformation   TriangleInformation  ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;

    static const int SMALL = 1;
    static const int LARGE = 0;

    void init();
    void reinit();
    void draw(const core::visual::VisualParams* vparams);
protected:
    TriangularAnisotropicFEMForceField();
    ~TriangularAnisotropicFEMForceField();
public:
    void computeMaterialStiffness(int i, Index& a, Index& b, Index& c);
    void getFiberDir(int element, Deriv& dir);

    //Data<Real> f_poisson2;
    //Data<Real> f_young2;
    Data<helper::vector<Real> > f_poisson2;
    Data<helper::vector<Real> > f_young2;
    Data<Real> f_theta;
    Data<VecCoord> f_fiberCenter;
    Data<bool> showFiber;

    topology::TriangleData <helper::vector< Deriv> > localFiberDirection;

    class TRQSTriangleHandler : public topology::TopologyDataHandler<topology::Triangle,vector<Deriv> >
    {
    public:
        typedef typename TriangularAnisotropicFEMForceField::Deriv triangleInfo;

        TRQSTriangleHandler(TriangularAnisotropicFEMForceField<DataTypes>* _ff, topology::TriangleData<helper::vector<triangleInfo> >*  _data) : topology::TopologyDataHandler<topology::Triangle, helper::vector<triangleInfo> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int triangleIndex,
                                 helper::vector<triangleInfo> & ,
                                 const topology::Triangle & t,
                                 const sofa::helper::vector< unsigned int > &,
                                 const sofa::helper::vector< double > &);

    protected:
        TriangularAnisotropicFEMForceField<DataTypes>* ff;
    };

    sofa::core::topology::BaseMeshTopology* _topology;

    TRQSTriangleHandler* triangleHandler;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API TriangularAnisotropicFEMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API TriangularAnisotropicFEMForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
