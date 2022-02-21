set(SOFAPRECONDITIONER_SRC src/SofaPreconditioner)

list(APPEND HEADER_FILES
    ${SOFAPRECONDITIONER_SRC}/BlockJacobiPreconditioner.h
    ${SOFAPRECONDITIONER_SRC}/BlockJacobiPreconditioner.inl
    ${SOFAPRECONDITIONER_SRC}/JacobiPreconditioner.h
    ${SOFAPRECONDITIONER_SRC}/JacobiPreconditioner.inl
    ${SOFAPRECONDITIONER_SRC}/PrecomputedWarpPreconditioner.h
    ${SOFAPRECONDITIONER_SRC}/PrecomputedWarpPreconditioner.inl
    ${SOFAPRECONDITIONER_SRC}/SSORPreconditioner.h
    ${SOFAPRECONDITIONER_SRC}/SSORPreconditioner.inl
    ${SOFAPRECONDITIONER_SRC}/WarpPreconditioner.h
    ${SOFAPRECONDITIONER_SRC}/WarpPreconditioner.inl
)
