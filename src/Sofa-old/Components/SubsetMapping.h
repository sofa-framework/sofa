#ifndef SOFA_COMPONENTS_SUBSETMAPPING_H
#define SOFA_COMPONENTS_SUBSETMAPPING_H

#include "Sofa-old/Core/MechanicalMapping.h"
#include "Sofa-old/Core/MechanicalModel.h"
#include "Sofa-old/Components/Common/DataField.h"
#include "Sofa-old/Components/Common/vector.h"
#include <vector>

namespace Sofa
{

namespace Components
{


/**
 * @class SubsetMapping
 * @brief Compute a subset of input points
 */
template <class BaseMapping>
class SubsetMapping : public BaseMapping
{
protected:
    /// Correspondance array
    typedef Common::vector<unsigned int> IndexArray;
    Common::DataField < IndexArray > f_indices;
    Common::DataField < int > f_first;
    Common::DataField < int > f_last;

public:
    typedef BaseMapping Inherit;
    typedef typename BaseMapping::In In;
    typedef typename BaseMapping::Out Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type Real;

    SubsetMapping(In* from, Out* to);

    void init();

    virtual ~SubsetMapping();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
};

} // namespace Components

} // namespace Sofa

#endif
