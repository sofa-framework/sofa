#ifndef INDEXVALUEMAPPER_H_
#define INDEXVALUEMAPPER_H_

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace engine
{


template <class DataTypes>
class IndexValueMapper : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IndexValueMapper,DataTypes),sofa::core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef unsigned int Index;

protected:

    IndexValueMapper();
    ~IndexValueMapper() {}
public:
    void init();
    void reinit();
    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const IndexValueMapper<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Input
    Data<sofa::helper::vector<Real> > f_inputValues;
    Data<sofa::helper::vector<Index> > f_indices;
    Data<Real> f_value;

    //Output
    Data<sofa::helper::vector<Real> > f_outputValues;

    //Parameter
    Data<Real> p_defaultValue;

};

#if defined(WIN32) && !defined(INDEXVALUEMAPPER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API IndexValueMapper<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API IndexValueMapper<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif


} // namespace engine

} // namespace component

} // namespace sofa

#endif /* INDEXVALUEMAPPER_H_ */
