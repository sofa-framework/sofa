import Vec

def id():
    return [0, 0, 0, 1]

def conj(q):
    return [-q[0], -q[1], -q[2], q[3]]

def inv(q):
    return Vec.scal(1 / Vec.norm2(q), conj(q) )

def re(q):
    return q[3]

def im(q):
    return q[:3]

# TODO optimize
def prod(a, b):
    return Vec.sum( Vec.sum( Vec.scal( re(a), im(b) ),
                             Vec.scal( re(b), im(a) )),
                    Vec.cross( im(a), im(b) ) ) + [ re(a) * re(b) - Vec.dot( im(a), im(b) ) ]

# TODO optimize
def rotate(q, x):
    # TODO assert q is unit
    return im( prod(q, prod( x + [0], conj(q))) )
