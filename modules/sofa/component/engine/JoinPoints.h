#ifndef SOFA_COMPONENT_ENGINE_JOINPOINTS_H_
#define SOFA_COMPONENT_ENGINE_JOINPOINTS_H_

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

/*
 * This engine join points within a given distance, merging into a new point which is the "average point".
 */

template <class DataTypes>
class JoinPoints : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(JoinPoints,DataTypes),sofa::core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;

protected:

    JoinPoints();
    ~JoinPoints() {}
public:
    void init();
    void reinit();
    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const JoinPoints<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Input
    Data<VecCoord > f_points;
    Data<Real> f_distance ;
    //Output
    Data<VecCoord > f_mergedPoints;



private:
    bool getNearestPoint(const typename std::list<Coord>::iterator &itCurrentPoint,
            std::list<Coord>& listPoints,
            std::list<int>& listCoeffs,
            typename std::list<Coord>::iterator &itNearestPoint,
            std::list<int>::iterator &itNearestCoeff,
            const Real& distance);

};

#if defined(WIN32) && !defined(JOINPOINTS_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API JoinPoints<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API JoinPoints<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_ENGINE_JOINPOINTS_H_ */
