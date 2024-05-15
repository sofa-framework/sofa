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
/******************************************************************************
* Contributors:
*   - "Nhan NGuyen" <nhnhanbk92@gmail.com> - JAIST (PRESTO Project)
*******************************************************************************/
#pragma once

#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

#include <map>
#include <sofa/helper/map.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class QuadBendingFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadBendingFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

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
    typedef sofa::core::topology::BaseMeshTopology::Quad Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads VecElement;
    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundVertex QuadsAroundVertex;

    typedef sofa::type::Quat<Real> Quat;

    enum {
        SMALL = 1,   ///< Symbol of large displacements quad solver
    };

protected:

    //bool _anisotropicMaterial;			                 	    /// used to turn on / off optimizations
    typedef type::Vec<20, Real> Displacement;					    ///< the displacement vector
    typedef type::Mat<8, 8, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef type::Mat<32, 20, Real> StrainDisplacement;				    ///< the strain-displacement matrix
    typedef type::Mat<20, 20, Real> Stiffness;					    ///< the stiffness matrix
    
protected:
    /// ForceField API
    QuadBendingFEMForceField();

    ~QuadBendingFEMForceField() override;
public:
    void init() override;
    void reinit() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;
/// Class to store FEM information on each quad, for topology modification handling
    class QuadInformation
    {
    public:
        /// material stiffness matrices of each quad including Bending and shear component
        MaterialStiffness BendingmaterialMatrix;
        MaterialStiffness ShearmaterialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        ///< the stiffness matrix
        Stiffness stiffness;
        Stiffness Bendingstiffness;
        Stiffness Shearstiffness;
        //Real area;
        // large displacement method
        type::fixed_array<Coord,3> InitialPosElements;
        Coord IntlengthElement;
        Coord IntheightElement;
        Coord Intcentroid;
        Real Inthalflength;
        Real Inthalfheight;

        //type::vector<Coord> lastNStressDirection;

        QuadInformation() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const QuadInformation& /*ti*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, QuadInformation& /*ti*/ )
        {
            return in;
        }

        Real differenceToCriteria;
    };
    
/// Class to store FEM information on each edge, for topology modification handling
    class EdgeInformation
    {
    public:
        EdgeInformation(){}

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

    /// Class to store FEM information on each vertex, for topology modification handling
    class VertexInformation
    {
    public:
        VertexInformation(){}

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const VertexInformation& /*vi*/)
        {
            return os;
        }
        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, VertexInformation& /*vi*/)
        {
            return in;
        }
    };

/// Topology Data
    core::topology::QuadData<sofa::type::vector<QuadInformation> > quadInfo;
    core::topology::PointData<sofa::type::vector<VertexInformation> > vertexInfo; ///< Internal point data
    core::topology::EdgeData<sofa::type::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    /** Method to initialize @sa QuadInformation when a new Quad is created.
    * Will be set as creation callback in the QuadData @sa quadInfo
    */
    void createQuadInformation(unsigned int quadIndex, QuadInformation&,
        const core::topology::BaseMeshTopology::Quad& t,
        const sofa::type::vector< unsigned int >&,
        const sofa::type::vector< SReal >&);

    sofa::core::topology::BaseMeshTopology* m_topology;
    
    /// Get/Set methods
    Real getPoisson() { return (f_poisson.getValue())[0]; }
    void setPoisson(Real val)
    {
        type::vector<Real> newP(1, val);
        f_poisson.setValue(newP);
    }
    Real getYoung() { return (f_young.getValue())[0]; }
    void setYoung(Real val)
    {
        type::vector<Real> newY(1, val);
        f_young.setValue(newY);
    }
    int  getMethod() { return method; }
    void setMethod(int val) { method = val; }
    void setMethod(const std::string& methodName); 
protected : 
    /// Forcefiled computations
    void computeDisplacementSmall(Displacement &D, Index elementIndex, const VecCoord &p);
    void computeBendingStrainDisplacement(StrainDisplacement &Jb, /*Index elementIndex,*/ Real gauss1, Real gauss2, Real l, Real h);
    void computeShearStrainDisplacement(StrainDisplacement &Js, /*Index elementIndex,*/ Real l, Real h);
    void computeElementStiffness( Stiffness &K, Index elementIndex);
    void computeForce(Displacement &F, Index elementIndex, Displacement &D);
    
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
    virtual void computeBendingMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d);
    virtual void computeShearMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d);
   
    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( VecCoord& f, const VecCoord & p, Index elementIndex);
    //void accumulateDampingSmall( VecCoord& f, Index elementIndex );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
    

public:

    /// Forcefield intern paramaters
    int method;
    Data<std::string> f_method; ///< large: large displacements, small: small displacements
    Data<type::vector<Real> > f_poisson; ///< Poisson ratio in Hooke's law (vector)
    Data<type::vector<Real> > f_young; ///< Young modulus in Hooke's law (vector)
    Data<Real> f_thickness; ///< Thickness of the elements

    /// Link to be set to the topology container in the component graph.
    SingleLink<QuadBendingFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    
};
#if !defined(SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP)

extern template class SOFA_MISC_FEM_API QuadBendingFEMForceField<defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP)

} // namespace sofa::component::solidmechanics::fem::elastic
