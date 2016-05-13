#ifndef PAIR_H
#define PAIR_H

#include <ostream>
#include <istream>


namespace sofa
{

namespace defaulttype
{

// TODO: why ?!

// and why does this not derive from pair ?!

    template<class First, class Second>
    struct SerializablePair
    {
        typedef SerializablePair<First,Second> SerializablePairType;
        typedef std::pair<First,Second> PairType;

        SerializablePair() {}
        SerializablePair( const SerializablePairType& p ) : pair(p.pair) {}
        SerializablePair( const PairType& p ) : pair(p) {}


        PairType pair;


        inline First& first() { return pair.first; }
        inline const First& first() const { return pair.first; }
        inline Second& second() { return pair.second; }
        inline const Second& second() const { return pair.second; }

        inline friend std::ostream& operator<<(std::ostream& out, const SerializablePairType& p)
        {
            return out << p.pair.first << " " << p.pair.second;
        }

        inline friend std::ostream& operator<<(std::ostream& out, const PairType& p)
        {
            return out << p.first << " " << p.second;
        }



        inline friend std::istream& operator>>(std::istream& in, SerializablePairType& p)
        {
            return in >> p.pair.first >> p.pair.second;
        }

        inline friend std::istream& operator>>(std::istream& in, PairType& p)
        {
            return in >> p.first >> p.second;
        }

        friend size_t hash_value(const SerializablePairType& c)
        {
            size_t hash = boost::hash<PairType>()(c.pair);
            return hash;
        }



    };






}
}

#endif
