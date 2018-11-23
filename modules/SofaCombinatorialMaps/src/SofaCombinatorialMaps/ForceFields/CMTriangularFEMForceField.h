/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_H_
#define SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_H_
#include <SofaCombinatorialMaps/config.h>

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <SofaCombinatorialMaps/BaseTopology/SurfaceTopologyContainer.h>
#include <SofaCombinatorialMaps/BaseTopology/CMTopologyData.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>

#include <SofaCombinatorialMaps/BaseTopology/SurfaceMaskTraversal.h>

#include <sofa/core/DataTracker.h>
#include <cgogn/core/utils/masks.h>

#include <functional>



#include <map>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace cm_forcefield
{


//#define PLOT_CURVE //lose some FPS


//using sofa::helper::vector;

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
class SOFA_COMBINATORIALMAPS_API CMTriangularFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(CMTriangularFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

	typedef unsigned int Index;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

	using SurfaceTopology = sofa::component::topology::SurfaceTopologyContainer;
	using BaseVertex = core::topology::CMapTopology::Vertex;
	using Vertex = SurfaceTopology::Vertex;
	using Edge = SurfaceTopology::Edge;
	using Face = SurfaceTopology::Face;
	using VecElement = SurfaceTopology::SeqHexahedra;

	using CellCache = SurfaceTopology::CellCache;
	using FilteredQuickTraversor = SurfaceTopology::FilteredQuickTraversor;

	template<typename T>
	using VertexAttribute = typename SurfaceTopology::Topology::template VertexAttribute<T>;
	template<typename T>
	using EdgeAttribute = typename SurfaceTopology::Topology::template EdgeAttribute<T>;
	template<typename T>
	using FaceAttribute = typename SurfaceTopology::Topology::template FaceAttribute<T>;

    typedef sofa::helper::Quater<Real> Quat;

    enum {
        LARGE = 0,   ///< Symbol of small displacements triangle solver
        SMALL = 1,   ///< Symbol of large displacements triangle solver
    };

protected:

    bool _anisotropicMaterial;			                 	    /// used to turn on / off optimizations
    typedef defaulttype::Vec<6, Real> Displacement;					    ///< the displacement vector
    typedef defaulttype::Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef sofa::helper::vector<MaterialStiffness> VecMaterialStiffness;   ///< a vector of material stiffness matrices
    typedef defaulttype::Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix
    typedef defaulttype::Mat<6, 6, Real> Stiffness;					    ///< the stiffness matrix
    typedef sofa::helper::vector<StrainDisplacement> VecStrainDisplacement; ///< a vector of strain-displacement matrices
    typedef defaulttype::Mat<3, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations

protected:
    /// ForceField API
    //{
	CMTriangularFEMForceField();

	virtual ~CMTriangularFEMForceField();
public:
    virtual void init() override;
    virtual void reinit() override;
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    void draw(const core::visual::VisualParams* vparams) override;
    //}

    /// Class to store FEM information on each triangle, for topology modification handling
    class TriangleInformation
    {
    public:
		/// the indices of the DOF (vertices) of the triangle
		helper::fixed_array<Index, 3> dofs;

        /// material stiffness matrices of each tetrahedron
        MaterialStiffness materialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        ///< the stiffness matrix
        Stiffness stiffness;
        Real area;
        // large displacement method
        helper::fixed_array<Coord,3> rotatedInitialElements;
        Transformation rotation;
        // strain vector
        defaulttype::Vec<3,Real> strain;
        // stress vector
        defaulttype::Vec<3,Real> stress;
        Transformation initialTransformation;
        Coord principalStressDirection;
        Real maxStress;
        Coord principalStrainDirection;
        Real maxStrain;

        helper::vector<Coord> lastNStressDirection;

        TriangleInformation() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInformation& /*ti*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInformation& /*ti*/ )
        {
            return in;
        }

        Real differenceToCriteria;
    };

    /// Class to store FEM information on each edge, for topology modification handling
    class EdgeInformation
    {
    public:
        EdgeInformation()
            :fracturable(false) {}

        bool fracturable;

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
        VertexInformation()
            :sumEigenValues(0.0), stress(0.0) {}

        Coord meanStrainDirection;
        double sumEigenValues;
        Transformation rotation;

        double stress; //average stress of triangles around (used only for drawing)

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
	//cm_topolgy::TriangleData<TriangleInformation> triangleInfo; ??
	FaceAttribute<TriangleInformation> triangleInfo;
	VertexAttribute<VertexInformation> vertexInfo;
	EdgeAttribute<EdgeInformation> edgeInfo;



	class TRQSTriangleHandler : public cm_topology::TopologyDataHandler<core::topology::CMapTopology::Face, TriangleInformation>
    {
    public:
		TRQSTriangleHandler(CMTriangularFEMForceField<DataTypes>* _ff,
							cm_topology::TriangleData<TriangleInformation>* _data)
			: cm_topology::TopologyDataHandler<core::topology::CMapTopology::Face, TriangleInformation >(_data)
			, ff(_ff)
		{}

		void applyCreateFunction(TriangleInformation& ,
				const core::topology::CMapTopology::Face & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

    protected:
	   CMTriangularFEMForceField<DataTypes>* ff;
    };

	SurfaceTopology* _topology;
	std::unique_ptr<FilteredQuickTraversor> cell_traversor;

    /// Get/Set methods
    Real getPoisson() { return (f_poisson.getValue())[0]; }
    void setPoisson(Real val)
    {
        helper::vector<Real> newP(1, val);
        f_poisson.setValue(newP);
    }
    Real getYoung() { return (f_young.getValue())[0]; }
    void setYoung(Real val)
    {
        helper::vector<Real> newY(1, val);
        f_young.setValue(newY);
    }
    Real getDamping() { return f_damping.getValue(); }
    void setDamping(Real val) { f_damping.setValue(val); }
    int  getMethod() { return method; }
    void setMethod(int val) { method = val; }
    void setMethod(std::string methodName) {
        if (methodName == "small")
            this->setMethod(SMALL);
        else
        {
            if (methodName != "large")
                serr << "unknown method: large method will be used. Remark: Available method are \"small\", \"large\" "<<sendl;
            this->setMethod(LARGE);
        }
    }


public:

    int  getFracturedEdge();

	void getFractureCriteria(TriangleInformation& info, Deriv& direction, Real& value);

    /// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
	void computeStressAlongDirection(Real &stress_along_dir, const TriangleInformation& info, const Coord &direction, const defaulttype::Vec<3,Real> &stress);

	/// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
	void computeStressAlongDirection(Real &stress_along_dir, Face face, const Coord &direction);

	/// Compute value of stress across a given direction (typically the fracture direction)
	void computeStressAcrossDirection(Real &stress_across_dir, Face face, const Coord &direction, const defaulttype::Vec<3,Real> &stress);

	/// Compute value of stress across a given direction (typically the fracture direction)
	void computeStressAcrossDirection(Real &stress_across_dir, Face face, const Coord &direction);

	/// Compute current stress
	void computeStress(defaulttype::Vec<3,Real> &stress, Face face);

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
	void getRotation(Transformation& R, Vertex nodeIdx);
    void getRotations();

protected :
    /// Forcefield computations
	void computeDisplacementSmall(Displacement &D, Face face, const VecCoord &p);
	void computeDisplacementLarge(Displacement &D, Face face, const Transformation &R_2_0, const VecCoord &p);
	void computeStrainDisplacement(StrainDisplacement& J, Face face, Coord a, Coord b, Coord c);
    void computeStiffness(StrainDisplacement &J, Stiffness &K, MaterialStiffness &D);
    void computeStrain(defaulttype::Vec<3,Real> &strain, const StrainDisplacement &J, const Displacement &D);
    void computeStress(defaulttype::Vec<3,Real> &stress, MaterialStiffness &K, defaulttype::Vec<3,Real> &strain);
	void computeForce(Displacement &F, Face face, const VecCoord &p);
	void computePrincipalStrain(TriangleInformation& info, defaulttype::Vec<3,Real>& strain);
	void computePrincipalStress(TriangleInformation& info);


    /// f += Kx where K is the stiffness matrix and x a displacement
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
	virtual void computeMaterialStiffness(TriangleInformation& info);

    ////////////// small displacements method
	void initSmall(Face f, const VecCoord& X0, TriangleInformation& info);
	void accumulateForceSmall(VecCoord& f, const VecCoord & p, Face face);
	void accumulateDampingSmall( VecCoord& f, Face face );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );

    ////////////// large displacements method
    //sofa::helper::vector< helper::fixed_array <Coord, 3> > _rotatedInitialElements;   ///< The initials positions in its frame
    //sofa::helper::vector< Transformation > _rotations;
	void initLarge(Face f, const VecCoord& X0, TriangleInformation& info);
	void computeRotationLarge(Transformation &r, const VecCoord &p, Face face);
	void accumulateForceLarge(VecCoord& f, const VecCoord & p, Face face);
	void accumulateDampingLarge( VecCoord& f, Face elementIndex );
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );


    bool updateMatrix;
    int lastFracturedEdgeIndex;

public:

    /// Forcefield intern paramaters
    int method;
    Data<std::string> f_method;
    Data<helper::vector<Real> > f_poisson;
    Data<helper::vector<Real> > f_young;
    Data<Real> f_damping;

	//TODO check cgogn function_traits that does not support functors
//    SingleLink<CMTriangularFEMForceField<DataTypes>, sofa::component::topology::SurfaceMaskTraversal, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mask;

    /// Initial strain parameters (if FEM is initialised with predefine values)
	typedef helper::fixed_array<Coord,3> RotatedInitialElements;
	using VecRotatedInitialElements = FaceAttribute<RotatedInitialElements>;
	Data<VecRotatedInitialElements> m_rotatedInitialElements;

	using VecTriangleTransformation = FaceAttribute<Transformation>;
	Data<VecTriangleTransformation> m_initialTransformation;

    /// Fracture parameters
    Data<bool> f_fracturable;
    Data<Real> hosfordExponant;
    Data<Real> criteriaValue;

    /// Display parameters
    Data<bool> showStressValue;
    Data<bool> showStressVector;
    Data<bool> showFracturableTriangles;

    Data<bool> f_computePrincipalStress;

    TRQSTriangleHandler* triangleHandler;

#ifdef PLOT_CURVE
    //structures to save values for each element along time
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphStress;
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphCriteria;
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphOrientation;

    //the index of element we want to display the graphs
    Data<Real>  elementID;

    //data storing the values along time for the element with index elementID
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphStress;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphCriteria;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphOrientation;
#endif

};


#if !defined(SOFA_COMPONENT_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_COMBINATORIALMAPS_API CMTriangularFEMForceField<defaulttype::Vec3dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_COMBINATORIALMAPS_API CMTriangularFEMForceField<defaulttype::Vec3fTypes>;
#endif

#endif // !defined(SOFA_COMPONENT_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_CPP)

} // namespace cm_forcefield

} // namespace component

} // namespace sofa

#endif // SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_H_
