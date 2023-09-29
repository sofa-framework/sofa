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
#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/TopologyData.h>


namespace sofa::component::solidmechanics::fem::elastic
{

namespace _beamfemforcefield_
{
using core::MechanicalParams;
using core::behavior::MultiMatrixAccessor;
using core::behavior::ForceField;
using core::topology::BaseMeshTopology;
using type::Vec;
using type::Mat;
using type::Vec3;
using type::Quat;
using core::topology::EdgeData;

/** Compute Finite Element forces based on 6D beam elements.
*/
template<class DataTypes>
class BeamFEMForceField : public ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BeamFEMForceField,DataTypes), SOFA_TEMPLATE(ForceField,DataTypes));

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;
    typedef VecCoord Vector;

    using Index = sofa::Index;

    typedef BaseMeshTopology::Edge Element;
    typedef type::vector<BaseMeshTopology::Edge> VecElement;
    
    typedef Vec<12, Real> Displacement;     ///< the displacement vector
    typedef Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations
    typedef Mat<12, 12, Real> StiffnessMatrix;

    struct BeamInfo
    {
        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
        SReal _E0,_E; //Young
        SReal _nu; //Poisson
        SReal _L; //length
        SReal _r; //radius of the section
        SReal _rInner; //inner radius of the section if beam is hollow
        SReal _G; //shear modulus
        SReal _Iy;
        SReal _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
        SReal _J;  //Polar moment of inertia (J = Iy + Iz)
        SReal _A; // A is the cross-sectional area;
        SReal _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
        SReal _Asz; //_Asz is the z-direction effective shear area;
        StiffnessMatrix _k_loc;

        type::Quat<SReal> quat;

        void init(SReal E, SReal L, SReal nu, SReal r, SReal rInner);

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const BeamInfo& bi )
        {
            os	<< bi._E0 << " "
                << bi._E << " "
                << bi._nu << " "
                << bi._L << " "
                << bi._r << " "
                << bi._rInner << " "
                << bi._G << " "
                << bi._Iy << " "
                << bi._Iz << " "
                << bi._J << " "
                << bi._A << " "
                << bi._Asy << " "
                << bi._Asz << " "
                << bi._k_loc;
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, BeamInfo& bi )
        {
            in	>> bi._E0
                >> bi._E
                >> bi._nu
                >> bi._L
                >> bi._r
                >> bi._rInner
                >> bi._G
                >> bi._Iy
                >> bi._Iz
                >> bi._J
                >> bi._A
                >> bi._Asy
                >> bi._Asz
                >> bi._k_loc;
            return in;
        }
    };

    EdgeData<type::vector<BeamInfo> > m_beamsData; ///< Internal element data

protected:
    void createBeamInfo(Index edgeIndex, BeamInfo&,
        const BaseMeshTopology::Edge& e,
        const type::vector<Index> &,
        const type::vector< SReal > &);

    const VecElement *m_indexedElements;

public:
    Data<Real> d_poissonRatio; ///< Poisson's Ratio
    Data<Real> d_youngModulus; ///< Young Modulus
    Data<Real> d_radius; ///< radius of the section
    Data<Real> d_radiusInner; ///< inner radius of the section for hollow beams
    Data< BaseMeshTopology::SetIndex > d_listSegment; ///< apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology
    Data< bool> d_useSymmetricAssembly; ///< use symmetric assembly of the matrix K

    /// Link to be set to the topology container in the component graph.
    SingleLink<BeamFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

 protected:
    //just for draw forces
    VecDeriv m_forces;

    bool m_partialListSegment;
    bool m_updateStiffnessMatrix;
    SReal m_lastUpdatedStep;

    Quat<SReal>& beamQuat(int i);

    BaseMeshTopology* m_topology;

    BeamFEMForceField();
    BeamFEMForceField(Real poissonRatio, Real youngModulus, Real radius, Real radiusInner);
    virtual ~BeamFEMForceField();

public:

    void init() override;
    void reinit() override;
    virtual void reinitBeam(Index i);
    void addForce(const MechanicalParams* mparams, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    void addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix ) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord&  x) const override;
    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void setUpdateStiffnessMatrix(bool val);
    void setBeam(Index i, SReal E, SReal L, SReal nu, SReal r, SReal rInner);
    void initBeams(std::size_t size);

protected:
    void drawElement(int i, std::vector< type::Vec3 >* points, const VecCoord& x);
    Real pseudoDeterminantForCoef ( const Mat<2, 3, Real>&  M );
    void computeStiffness(int i, Index a, Index b);

    /// Large displacements method
    type::vector<Transformation> _nodeRotations;
    void initLarge(int i, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b, SReal fact=1.0);
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BeamFEMForceField<defaulttype::Rigid3Types>;
#endif

} // namespace _beamfemforcefield_

using _beamfemforcefield_::BeamFEMForceField;

} // namespace sofa::component::solidmechanics::fem::elastic
