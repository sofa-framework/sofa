#ifndef SOFA_COMPONENT_MAPPING_SUBSETMAPPING_H
#define SOFA_COMPONENT_MAPPING_SUBSETMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/DataField.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/**
 * @class SubsetMapping
 * @brief Compute a subset of input points
 */
template <class BasicMapping>
class SubsetMapping : public BasicMapping
{
protected:
    /// Correspondance array
    typedef helper::vector<unsigned int> IndexArray;
    DataField < IndexArray > f_indices;
    DataField < int > f_first;
    DataField < int > f_last;

public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
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

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
