
#ifndef _BASE_H_
#define _BASE_H_


inline uint log_2 ( uint a )  // returns [log_2 a]+1
{
    uint res = 0 ;
    for ( ; a != 0 ; a >>=1, res++ ) ;
    return res ;
}

#endif // _BASE_H_
