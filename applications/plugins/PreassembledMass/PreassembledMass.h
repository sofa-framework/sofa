#ifndef __PreassembledMass_H
#define __PreassembledMass_H


#include "initPlugin.h"
#include <sofa/core/behavior/Mass.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>


#if SOFA_HAVE_FLEXIBLE
#include <plugins/Flexible/types/AffineTypes.h>
#include <plugins/Flexible/types/QuadraticTypes.h>
#endif


namespace sofa
{

namespace component
{

namespace mass
{



/** Not templated class for PreassembledMass */
class SOFA_PreassembledMass_API BasePreassembledMass
{
protected:
    static unsigned int s_instanciationCounter; ///< how many PreassembledMass components?
};


/**
* Precompute mass at this level (performing an assembly of the mapped masses), then destroying the child mass components.
@warning all independent dofs must be in independent graph branches (ie the XML technique for multimappings is not applicable)
@warning all the parents of a mapped mass must be preassembled (since the mapped mass will be deleted)
*/
template <class DataTypes>
class PreassembledMass : public core::behavior::Mass< DataTypes >, public BasePreassembledMass
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PreassembledMass,DataTypes), SOFA_TEMPLATE( core::behavior::Mass, DataTypes ) );



    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef SReal Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef component::linearsolver::EigenSparseMatrix<DataTypes,DataTypes> MassMatrix;


protected:

    PreassembledMass()
        : core::behavior::Mass<DataTypes>()
        , d_massMatrix( initData(&d_massMatrix, "massMatrix", "AssembledMassMatrix") )
        , l_massNodes(initLink("massNodes", "Nodes to deactivate (that were only mapping a mass)"))
    {
        _instanciationNumber = s_instanciationCounter++;
    }

    virtual ~PreassembledMass()
    {
        --s_instanciationCounter;
    }

    unsigned int _instanciationNumber; ///< its PreassembledMass index




public:

    virtual void init();

    virtual void bwdInit();

    static std::string templateName(const PreassembledMass<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
	virtual std::string getTemplateName() const
	{
		return templateName(this);
	}


    // -- Mass interface (employed only if d_massOnIndependents==true)
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, double factor);
    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);
    double getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const;
    double getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;
    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v);
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);





    Data<MassMatrix> d_massMatrix; ///< assembled mass matrix





    typedef MultiLink<PreassembledMass<DataTypes>,core::objectmodel::BaseContext,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DOUBLELINK> LinkMassNodes;
    LinkMassNodes l_massNodes;
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(__PreassembledMass_CPP)

extern template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Vec3Types>; // volume FEM (tetra, hexa)
extern template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Vec1Types>; // subspace
extern template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Rigid3Types>; // rigid frames

#if SOFA_HAVE_FLEXIBLE
extern template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Affine3Types>; // affine frames
#endif

#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif // __PreassembledMass_H
