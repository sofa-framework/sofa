#pragma once

#include <string_view>

namespace sofa::type
{

template<class T> struct TypeTrait{};

#define MAKE_TYPE_TRAIT(type, suffix_string) \
    template<> struct TypeTrait<type> \
    { \
        static constexpr std::string_view typeName = #type; \
        static constexpr std::string_view suffix = suffix_string; \
        static const std::string name() { return std::string(typeName); } \
        static const std::string GetTypeName() { return std::string(suffix); } \
    }

MAKE_TYPE_TRAIT(bool, "bool");

MAKE_TYPE_TRAIT(float, "f");
MAKE_TYPE_TRAIT(double, "d");
MAKE_TYPE_TRAIT(long double, "e");

MAKE_TYPE_TRAIT(int, "i");
MAKE_TYPE_TRAIT(unsigned int, "I");

MAKE_TYPE_TRAIT(short, "h");
MAKE_TYPE_TRAIT(unsigned short, "H");

MAKE_TYPE_TRAIT(char, "b");
MAKE_TYPE_TRAIT(unsigned char, "B");

MAKE_TYPE_TRAIT(long, "l");
MAKE_TYPE_TRAIT(unsigned long, "L");

MAKE_TYPE_TRAIT(long long, "q");
MAKE_TYPE_TRAIT(unsigned long long, "Q");

#undef MAKE_TYPE_TRAIT

}
