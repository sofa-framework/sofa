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
#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H


#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include "config.h"


namespace sofa
{

namespace component
{

namespace container
{
class StiffnessContainer;
class PoissonContainer;
} // namespace container

namespace forcefield
{

namespace _beamfemforcefield_ {

using topology::TopologyDataHandler;
using helper::vector;
using core::MechanicalParams;
using core::behavior::MultiMatrixAccessor;
using core::behavior::ForceField;
using core::topology::BaseMeshTopology;
using defaulttype::Vec;
using defaulttype::Mat;
using defaulttype::Vector3;
using defaulttype::Quat;
using topology::EdgeData;

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

    typedef unsigned int Index;
    typedef BaseMeshTopology::Edge Element;
    typedef vector<BaseMeshTopology::Edge> VecElement;
    typedef vector<unsigned int> VecIndex;
    typedef Vec<3, Real> Vec3;

protected:

    typedef Vec<12, Real> Displacement;     ///< the displacement vector
    typedef Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations
    typedef Mat<12, 12, Real> StiffnessMatrix;

    struct BeamInfo
    {
        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
        double _E0,_E; //Young
        double _nu; //Poisson
        double _L; //length
        double _r; //radius of the section
        double _rInner; //inner radius of the section if beam is hollow
        double _G; //shear modulus
        double _Iy;
        double _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
        double _J;  //Polar moment of inertia (J = Iy + Iz)
        double _A; // A is the cross-sectional area;
        double _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
        double _Asz; //_Asz is the z-direction effective shear area;
        StiffnessMatrix _k_loc;

        defaulttype::Quat quat;

        void init(double E, double L, double nu, double r, double rInner);

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

    class BeamFFEdgeHandler : public TopologyDataHandler<BaseMeshTopology::Edge, vector<BeamInfo> >
    {
    public:
        typedef typename BeamFEMForceField<DataTypes>::BeamInfo BeamInfo;
        BeamFFEdgeHandler(BeamFEMForceField<DataTypes>* ff, EdgeData<vector<BeamInfo> >* data)
            :TopologyDataHandler<BaseMeshTopology::Edge, vector<BeamInfo> >(data),ff(ff) {}

        void applyCreateFunction(unsigned int edgeIndex, BeamInfo&,
                                 const BaseMeshTopology::Edge& e,
                                 const vector<unsigned int> &,
                                 const vector< double > &);

    protected:
        BeamFEMForceField<DataTypes>* ff;

    };

    //just for draw forces
    VecDeriv m_forces;
    EdgeData<vector<BeamInfo> > m_beamsData; ///< Internal element data
    linearsolver::EigenBaseSparseMatrix<typename DataTypes::Real> m_matS;

    const VecElement *m_indexedElements;
    Data<Real> d_poissonRatio; ///< Potion Ratio
    Data<Real> d_youngModulus; ///< Young Modulus
    Data<Real> d_radius; ///< radius of the section
    Data<Real> d_radiusInner; ///< inner radius of the section for hollow beams
    Data< VecIndex > d_listSegment; ///< apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology
    Data< bool> d_useSymmetricAssembly; ///< use symmetric assembly of the matrix K
    bool m_partialListSegment;
    bool m_updateStiffnessMatrix;
    bool m_assembling;
    double m_lastUpdatedStep;

    container::StiffnessContainer* m_stiffnessContainer;
    container::PoissonContainer* m_poissonContainer;

    Quat& beamQuat(int i);

    BaseMeshTopology* m_topology;
    BeamFFEdgeHandler* m_edgeHandler;

    BeamFEMForceField();
    BeamFEMForceField(Real poissonRatio, Real youngModulus, Real radius, Real radiusInner);
    virtual ~BeamFEMForceField();

public:

    virtual void init() override;
    virtual void bwdInit() override;
    virtual void reinit() override;
    virtual void reinitBeam(unsigned int i);
    virtual void addForce(const MechanicalParams* mparams, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    virtual void addDForce(const MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    virtual void addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix ) override;
    virtual SReal getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord&  x) const override;
    virtual void draw(const core::visual::VisualParams* vparams) override;
    virtual void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void setUpdateStiffnessMatrix(bool val);
    void setComputeGlobalMatrix(bool val);
    void setBeam(unsigned int i, double E, double L, double nu, double r, double rInner);
    void initBeams(unsigned int size);

protected:

    void drawElement(int i, std::vector< Vector3 >* points, const VecCoord& x);
    Real pseudoDeterminantForCoef ( const Mat<2, 3, Real>&  M );
    void computeStiffness(int i, Index a, Index b);

    /// Large displacements method
    vector<Transformation> _nodeRotations;
    void initLarge(int i, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b, double fact=1.0);
};

#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3fTypes>;
#endif

}

using _beamfemforcefield_::BeamFEMForceField;

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
