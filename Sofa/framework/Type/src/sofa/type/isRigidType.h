//
// Created by hugo on 28/11/23.
//

#ifndef SOFA_ISRIGIDTYPE_H
#define SOFA_ISRIGIDTYPE_H

namespace sofa::type
{
    // Boiler-plate code to test if a type implements a method
    // explanation https://stackoverflow.com/a/30848101

    template <typename...>
    using void_t = void;

    // Primary template handles all types not supporting the operation.
    template <typename, template <typename> class, typename = void_t<>>
    struct detect : std::false_type {};

    // Specialization recognizes/validates only types supporting the archetype.
    template <typename T, template <typename> class Op>
    struct detect<T, Op, void_t<Op<T>>> : std::true_type {};

    // Actual test if DataType::Coord implements getOrientation() (hence is a RigidType)
    template <typename T>
    using isRigid_t = decltype(std::declval<typename T::Coord>().getOrientation());

    template <typename T>
    using isRigidType = detect<T, isRigid_t>;
}
#endif //SOFA_ISRIGIDTYPE_H
