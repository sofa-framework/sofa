/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>


namespace sofa
{

namespace component
{


namespace forcefield
{


template<class DataTypes>
class TetrahedralTensorMassForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedralTensorMassForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename VecCoord::template rebind<VecCoord>::other VecType;


    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    class Mat3 : public helper::fixed_array<Deriv,3>
    {
    public:
        Deriv operator*(const Deriv& v) const
        {
            return Deriv((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
        Deriv transposeMultiply(const Deriv& v) const
        {
            return Deriv(v[0]*((*this)[0])[0]+v[1]*((*this)[1])[0]+v[2]*((*this)[2][0]),
                    v[0]*((*this)[0][1])+v[1]*((*this)[1][1])+v[2]*((*this)[2][1]),
                    v[0]*((*this)[0][2])+v[1]*((*this)[1][2])+v[2]*((*this)[2][2]));
        }
    };

protected:

    class EdgeRestInformation
    {
    public:
        Mat3 DfDx; /// the edge stiffness matrix
        float vertices[2]; // the vertices of this edge

        EdgeRestInformation()
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeRestInformation& /*eri*/ )
        {
            return in;
        }
    };
    typedef typename VecCoord::template rebind<EdgeRestInformation>::other edgeRestInfoVector;


    sofa::core::topology::BaseMeshTopology* _topology;
    VecCoord  _initialPoints;///< the intial positions of the points

    bool updateMatrix;

    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient

    TetrahedralTensorMassForceField();

    virtual ~TetrahedralTensorMassForceField();

public:

    virtual void init();
    void initNeighbourhoodPoints();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams) const ;
    void setYoungModulus(const double modulus)
    {
        f_youngModulus.setValue((Real)modulus);
    }
    void setPoissonRatio(const double ratio)
    {
        f_poissonRatio.setValue((Real)ratio);
    }
    void draw(const core::visual::VisualParams* vparams);
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();


    class TetrahedralTMEdgeHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,edgeRestInfoVector >
    {
    public:
        typedef typename TetrahedralTensorMassForceField<DataTypes>::EdgeRestInformation EdgeRestInformation;
        TetrahedralTMEdgeHandler(TetrahedralTensorMassForceField<DataTypes>* _ff, topology::EdgeData<edgeRestInfoVector >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, edgeRestInfoVector >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int edgeIndex, EdgeRestInformation& ei,
                const core::topology::BaseMeshTopology::Edge &,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

        void applyTetrahedronCreation(const sofa::helper::vector<unsigned int> &edgeAdded,
                const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &,
                const sofa::helper::vector<sofa::helper::vector<unsigned int> > &,
                const sofa::helper::vector<sofa::helper::vector<double> > &);

        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> &edgeRemoved);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,edgeRestInfoVector >::ApplyTopologyChange;
        /// Callback to add tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);

    protected:
        TetrahedralTensorMassForceField<DataTypes>* ff;
    };

protected:

//    EdgeData < typename VecType < EdgeRestInformation > > edgeInfo;
    topology::EdgeData < edgeRestInfoVector > edgeInfo;

//    EdgeData < typename VecType < EdgeRestInformation > > &getEdgeInfo() {return edgeInfo;}
    topology::EdgeData < edgeRestInfoVector > &getEdgeInfo() {return edgeInfo;}


    TetrahedralTMEdgeHandler* edgeHandler;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_MISC_FEM_API TetrahedralTensorMassForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_FEM_API TetrahedralTensorMassForceField<sofa::defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)


} //namespace forcefield

} // namespace component

} // namespace sofa



#endif /* _TetrahedralTensorMassForceField_H_ */
