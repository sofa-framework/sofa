#ifndef SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_H
#define SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/ForceField.h>

namespace sofa
{
namespace constraint
{

template <class DataTypes>
class MappingGeometricStiffnessForceField final : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MappingGeometricStiffnessForceField, DataTypes), 
               SOFA_TEMPLATE(sofa::core::behavior::ForceField,DataTypes) );
    
    typedef Inherit1 Inherit;
    typedef sofa::SingleLink< MyType, sofa::core::BaseMapping, 
        sofa::BaseLink::FLAG_STRONGLINK | sofa::BaseLink::FLAG_STOREPATH > MappingLink;
    typedef typename Inherit::DataVecDeriv DataVecDeriv;
    typedef typename Inherit::DataVecCoord DataVecCoord;

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    //template<class MatrixWriter>
    //void addKToMatrixT(const sofa::core::MechanicalParams* mparams, MatrixWriter m);

    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&) const override
    {
        return 0;
    }

protected:
    MappingGeometricStiffnessForceField();

    ~MappingGeometricStiffnessForceField();

private:
    MappingLink l_mapping;
};




}

}


#endif // ! SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_H
