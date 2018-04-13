/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Interface: TriangularBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <map>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.


	@author The SOFA team </www.sofa-framework.org>
*/
template<class DataTypes>
class TriangularBendingSprings : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularBendingSprings, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    //typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    //typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

protected:

    //Data<double> ks;
    //Data<double> kd;

    class EdgeInformation
    {
    public:
        Mat DfDx; /// the edge stiffness matrix

        int     m1, m2;  /// the two extremities of the spring: masses m1 and m2

        double  ks;      /// spring stiffness (initialized to the default value)
        double  kd;      /// damping factor (initialized to the default value)

        double  restlength; /// rest length of the spring

        bool is_activated;

        bool is_initialized;

        EdgeInformation(int m1=0, int m2=0, /* double ks=getKs(), double kd=getKd(), */ double restlength=0.0, bool is_activated=false, bool is_initialized=false)
            : m1(m1), m2(m2), /* ks(ks), kd(kd), */ restlength(restlength), is_activated(is_activated), is_initialized(is_initialized)
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::component::topology::EdgeData<helper::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    class TriangularBSEdgeHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >
    {
    public:
        typedef typename TriangularBendingSprings<DataTypes>::EdgeInformation EdgeInformation;
        TriangularBSEdgeHandler(TriangularBendingSprings<DataTypes>* _ff, topology::EdgeData<helper::vector<EdgeInformation> >* _data)
            : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int edgeIndex,
                EdgeInformation &ei,
                const core::topology::BaseMeshTopology::Edge& ,  const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);

        void applyTriangleCreation(const helper::vector<unsigned int> &triangleAdded,
                const helper::vector<core::topology::BaseMeshTopology::Triangle> & ,
                const helper::vector<helper::vector<unsigned int> > & ,
                const helper::vector<helper::vector<double> > &);

        void applyTriangleDestruction(const helper::vector<unsigned int> &triangleRemoved);

        void applyPointDestruction(const helper::vector<unsigned int> &pointIndices);

        void applyPointRenumbering(const helper::vector<unsigned int> &pointToRenumber);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >::ApplyTopologyChange;
        /// Callback to add triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);

        /// Callback to remove points elements.
        void ApplyTopologyChange(const core::topology::PointsRemoved* /*event*/);
        /// Callback to renumbering on points elements.
        void ApplyTopologyChange(const core::topology::PointsRenumbering* /*event*/);

    protected:
        TriangularBendingSprings<DataTypes>* ff;
    };

    sofa::core::topology::BaseMeshTopology* _topology;

    bool updateMatrix;

    Data<double> f_ks; ///< uniform stiffness for the all springs
    Data<double> f_kd; ///< uniform damping for the all springs



    TriangularBendingSprings(/*double _ks, double _kd*/);
    //TriangularBendingSprings(); //MechanicalState<DataTypes> *mm1 = NULL, MechanicalState<DataTypes> *mm2 = NULL);

    virtual ~TriangularBendingSprings();
public:
    /// Searches triangle topology and creates the bending springs
    virtual void init() override;

    virtual void reinit() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& d_x) const override;

    virtual double getKs() const { return f_ks.getValue();}
    virtual double getKd() const { return f_kd.getValue();}

    void setKs(const double ks)
    {
        f_ks.setValue((double)ks);
    }
    void setKd(const double kd)
    {
        f_kd.setValue((double)kd);
    }

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    sofa::component::topology::EdgeData<helper::vector<EdgeInformation> > &getEdgeInfo() {return edgeInfo;}

    TriangularBSEdgeHandler* edgeHandler;

    SReal m_potentialEnergy;

    //public:
    //Data<double> ks;
    //Data<double> kd;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_DEFORMABLE_API TriangularBendingSprings<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_DEFORMABLE_API TriangularBendingSprings<defaulttype::Vec3fTypes>;
#endif
#endif //defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_H
