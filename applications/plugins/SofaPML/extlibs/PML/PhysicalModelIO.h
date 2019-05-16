// Allow to compile the stl types without warnings on MSVC
// this line should be BEFORE the #ifndef
//#ifdef WIN32
//#pragma warning(disable:4786) // disable C4786 warning
//#endif


// Fucking std io C ..
#ifndef __PHYSICALMODELIO_H_
#define __PHYSICALMODELIO_H_

 
#include <string> 
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <exception>
#include <sstream>

// using hash_map or not, depending on the plateform
// (hash_map is much powerful, but it is not defined in
// MSVC++). hash_map is available in the SGI STL implementation
// which is the one defined in gcc...
//#ifdef WIN32
#include <map>
#include <set>
// from now on, everything written impmap should be understand map
#define impmap std::map
#define impset std::set
/*#else
// include the file
#include <hash_map>
#include <hash_set>
// from now on, everything written impmap should be understand hash_map
#define impmap std::hash_map
#define impset std::hash_set

/// special hash code functions (to be able to use a hash_map and hash_set)

/// PM component
class Component;
struct hash<Component *> {
  size_t operator()(const Component * cp) const { return (size_t) cp; } // simply return the address
};
class Atom;
struct hash<Atom *> {
  size_t operator()(const Atom * a) const { return (size_t) a; } // simply return the address
};

#endif
*/

#endif
