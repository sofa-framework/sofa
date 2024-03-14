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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

#include <sofa/type/trait/Rebind.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class TriangularFEMForceFieldOptim;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class TriangularFEMForceFieldOptimInternalData
{
public:
    typedef TriangularFEMForceFieldOptim<DataTypes> Main;
    void reinit(Main* /*m*/) {}
};


/** corotational triangle from
* @InProceedings{NPF05,
*   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Efficient, Physically Plausible Finite Elements",
*   booktitle    = "Eurographics (short papers)",
*   month        = "august",
*   year         = "2005",
*   editor       = "J. Dingliana and F. Ganovelli",
*   keywords     = "animation, physical model, elasticity, finite elements",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
* }
*/

template<class DataTypes>
class TriangularFEMForceFieldOptim : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularFEMForceFieldOptim, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Index Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;
    typedef sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;

    typedef sofa::type::Quat<Real> Quat;

protected:
    typedef type::Mat<2, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations
    typedef type::Mat<3, 3, Real> MaterialStiffness;
    enum { DerivSize = DataTypes::deriv_total_size };
    typedef type::Mat<DerivSize, DerivSize, Real> MatBloc;

    typedef TriangularFEMForceFieldOptimInternalData<DataTypes> InternalData;
    InternalData data;

protected:
    /// ForceField API
    TriangularFEMForceFieldOptim();

    virtual ~TriangularFEMForceFieldOptim();
public:
    Real getPoisson() { return d_poisson.getValue(); }
    Real getYoung() { return d_young.getValue(); }

    void init() override;
    void reinit() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;
    
    void computePrincipalStress();
    void getTrianglePrincipalStress(Index i, Real& stressValue, Deriv& stressDirection);

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;
    void draw(const core::visual::VisualParams* vparams) override;

    // parse method attribute (for compatibility with non-optimized version)
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Class to store FEM information on each triangle, for topology modification handling
    class TriangleInfo
    {
    public:
        //Index ia, ib, ic;
        Real bx, cx, cy, ss_factor;
        Transformation init_frame; // Mat<2,3,Real>

        Real stress;
        Deriv stressVector;
        Real stress2;
        Deriv stressVector2;

        TriangleInfo() :bx(0), cx(0), cy(0), ss_factor(0) { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInfo& ti )
        {
            return os << "bx= " << ti.bx << " cx= " << ti.cx << " cy= " << ti.cy << " ss_factor= " << ti.ss_factor << " init_frame= " << ti.init_frame << " END";
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInfo& ti )
        {
            std::string str;
            while (in >> str)
            {
                if (str == "END") break;
                else if (str == "bx=") in >> ti.bx;
                else if (str == "cx=") in >> ti.cx;
                else if (str == "cy=") in >> ti.cy;
                else if (str == "ss_factor=") in >> ti.ss_factor;
                else if (str == "init_frame=") in >> ti.init_frame;
                else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
            }
            return in;
        }
    };

    Real gamma, mu;
    class TriangleState
    {
    public:
        Transformation frame; // Mat<2,3,Real>
        Deriv stress;

        TriangleState() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleState& ti )
        {
            return os << "frame= " << ti.frame << " stress= " << ti.stress << " END";
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleState& ti )
        {
            std::string str;
            while (in >> str)
            {
                if (str == "END") break;
                else if (str == "frame=") in >> ti.frame;
                else if (str == "stress=") in >> ti.stress;
                else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
            }
            return in;
        }
    };

    /// Topology Data
    using VecTriangleInfo  = sofa::type::rebind_to<VecCoord, TriangleInfo>;
    using VecTriangleState = sofa::type::rebind_to<VecCoord, TriangleState>;

    core::topology::TriangleData<VecTriangleInfo> d_triangleInfo; ///< Internal triangle data (persistent)
    core::topology::TriangleData<VecTriangleState> d_triangleState; ///< Internal triangle data (time-dependent)

    /** Method to create @sa TriangleInfo when a new triangle is created.
    * Will be set as creation callback in the TriangleData @sa d_triangleInfo
    */
    void createTriangleInfo(Index triangleIndex, TriangleInfo&, 
        const Triangle& t,
        const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /** Method to create @sa TriangleState when a new triangle is created.
    * Will be set as creation callback in the TriangleData @sa d_triangleState
    */
    void createTriangleState(Index triangleIndex, TriangleState&, 
        const Triangle& t,
        const sofa::type::vector< Index > &,
        const sofa::type::vector< SReal > &);

    void initTriangleInfo(Index triangleIndex, TriangleInfo& ti, const Triangle t, const VecCoord& x0);
    void initTriangleState(Index triangleIndex, TriangleState& ti, const Triangle t, const VecCoord& x);

    void computeTriangleRotation(Transformation& result, Coord eab, Coord eac);
    void computeTriangleRotation(Transformation& result, Coord a, Coord b, Coord c)
    {
        computeTriangleRotation(result,b-a,c-a);
    }
    void computeTriangleRotation(Transformation& result, VecCoord& x0, Triangle t)
    {
        computeTriangleRotation(result,x0[t[0]], x0[t[1]], x0[t[2]]);
    }

    void getTriangleVonMisesStress(Index i, Real& stressValue);
    void getTrianglePrincipalStress(Index i, Real& stressValue, Deriv& stressDirection, Real& stressValue2, Deriv& stressDirection2);

    /// Public methods to access FEM information per element. Those method should not be used internally as they add check on element id.
    type::fixed_array <Coord, 3> getRotatedInitialElement(Index elemId);
    Transformation getRotationMatrix(Index elemId);
    MaterialStiffness getMaterialStiffness(Index elemId);
    type::Vec3 getStrainDisplacementFactors(Index elemId);
    Real getTriangleFactor(Index elemId);

public:

    /// Forcefield intern paramaters
    Data<Real> d_poisson;
    Data<Real> d_young; ///< Young modulus in Hooke's law
    Data<Real> d_damping; ///< Ratio damping/stiffness
    Data<Real> d_restScale; ///< Scale factor applied to rest positions (to simulate pre-stretched materials)

    Data<bool> d_computePrincipalStress; ///< Compute principal stress for each triangle
    Data<Real> d_stressMaxValue; ///< Max stress value computed over the triangulation

    /// Display parameters
    Data<bool> d_showStressVector; ///< Flag activating rendering of stress directions within each triangle
    Data<Real> d_showStressThreshold; ///< Minimum Stress value for rendering of stress vectors

    /// Link to be set to the topology container in the component graph. 
    SingleLink<TriangularFEMForceFieldOptim<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    /// Pointer to the topology container. Will be set by link @sa l_topology
    sofa::core::topology::BaseMeshTopology* m_topology;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TriangularFEMForceFieldOptim<defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_CPP)

} // namespace sofa::component::solidmechanics::fem::elastic
