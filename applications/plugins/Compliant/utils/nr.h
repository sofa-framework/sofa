#ifndef COMPLIANT_UTILS_NR_H
#define COMPLIANT_UTILS_NR_H

#include <cmath>

namespace utils {

namespace nr {


// void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb, float *fc, float (*func)(float)) 
// {
//     float ulim,u,r,q,fu,dum;
//     *fa=(*func)(*ax); *fb=(*func)(*bx);
//     if (*fb > *fa) {
//         SHFT(dum,*ax,*bx,dum);
//         SHFT(dum,*fb,*fa,dum);
//     }
//     *cx=(*bx)+GOLD*(*bx-*ax);
//     *fc=(*func)(*cx);
    
//     while (*fb > *fc) {
//         r=(*bx-*ax)*(*fb-*fc);
//         q=(*bx-*cx)*(*fb-*fa);
//         u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
//             (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
//         ulim=(*bx)+GLIMIT*(*cx-*bx);

        
//         if ((*bx-u)*(u-*cx) > 0.0) {
//             fu=(*func)(u);
            
//             if (fu < *fc) {
//                 *ax=(*bx);
//                 *bx=u;
//                 *fa=(*fb);
//                 *fb=fu;
//                 return;
//             } else if (fu > *fb) {
//                 *cx=u;
//                 *fc=fu;
//                 return;

//             }
//             u=(*cx)+GOLD*(*cx-*bx);
//             fu=(*func)(u);
//         } else if ((*cx-u)*(u-ulim) > 0.0) {
//             fu=(*func)(u);
//             if (fu < *fc) {
//                 SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx));
//                 SHFT(*fb,*fc,fu,(*func)(u));
//             }
//         } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
//             u=ulim;
//             fu=(*func)(u);
//         } else {
//             u=(*cx)+GOLD*(*cx-*bx);
//             fu=(*func)(u);
//         }

//         SHFT(*ax,*bx,*cx,u);
//         SHFT(*fa,*fb,*fc,fu);
//     }
// }

template<class U>
struct optimization {

    struct func_call {
        U x, f;
    };

    static const U gold;
    static const U tiny;

    template<class F>
    static void minimum_bracket(func_call& a, func_call& b, func_call& c,
                                const F& f) {

        using namespace std;
        
        // maximum magnification
        static const U glimit = 100;

        func_call u;

        a.f = f(a.x);
        b.f = f(b.x);

        if (b.f > a.f) {
            swap(a, b);
        }

        c.x = b.x + gold * (b.x - a.x);
        c.f = f(c.x);
        
        while (b.f > c.f) {
            const U r = (b.x - a.x) * (b.f - c.f);
            const U q = (b.x - c.x) * (b.f - a.f);

            u.x = b.x - ((b.x - c.x) * q - (b.x - a.x) * r ) /
                (2 * copysign(max(abs(q - r), tiny),
                                   q - r) );
            
            const U ulim = b.x + glimit * (c.x - b.x);
            
            if( (b.x - u.x) * (u.x - c.x) > 0 ) {
                u.f = f(u.x);
                
                if (u.f < c.f) {
                    a = b;
                    b = u;
                    return;
                } else if (u.f > b.f) {
                    c = u;
                    return;
                }
                
                u.x = c.x + gold * (c.x - b.x);
                u.f = f(u.x);
                
            } else if ((c.x - u.x) * (u.x - ulim) > 0) {
                u.f = f(u.x);
                
                if (u.f < c.f) {

                    b = c;
                    c = u;

                    // the tricky one
                    u.x = c.x + gold * (c.x - b.x);
                    u.f = f(u.x);
                    
                }
                
            } else if ( (u.x - ulim) * (ulim - c.x) >= 0) {
                u.x = ulim;
                u.f = f(u.x);
            } else {
                u.x = c.x + gold * (c.x - b.x);
                u.f = f(u.x);
            }

            a = b;
            b = c;
            c = u;
            
        }
        
    }
    
};

template<class U>
const U optimization<U>::gold = 1.618033988;

template<class U>
const U optimization<U>::tiny = 1e-20;



}





}


#endif
