# Estimate the randoms contribution r_e for each coincidence from the singles.
# Random coincidences between detector elements i,j occur at R_ij = 2τ S_i S_j / T
# (S = singles per element over the acquisition, τ = coincidence window, T =
# acquisition time). We histogram the singles onto the (iz,iphi) element grid to
# get S, then evaluate r for EVERY event (trues included — the randoms rate exists
# on every LOR, it is not a per-event label). The S_i·S_j product is the spatial
# shape; the absolute scale comes from T, or from calibrating the total to a known
# random count (we have the truth flags to anchor it).

using HDF5

"Linear element id, matching PTCRYSP block_id = iz·n_phi + iphi (0-based), as a 1-based index."
elem_id(iz, iphi, n_phi) = Int(iz) * Int(n_phi) + Int(iphi) + 1

"""
    singles_element_counts(path; n_phi, n_z) -> S

Histogram the singles in a PTCRYSP `singles.h5` onto the `n_phi*n_z` detector
elements: `S[elem_id(iz,iphi)]` is the number of singles on that element.
"""
function singles_element_counts(path::AbstractString; n_phi::Integer, n_z::Integer)
    S = zeros(Float64, Int(n_phi) * Int(n_z))
    h5open(path, "r") do f
        iz = read(f["iz"]); iphi = read(f["iphi"])
        @inbounds for k in eachindex(iz)
            S[elem_id(iz[k], iphi[k], n_phi)] += 1.0
        end
    end
    return S
end

"""
    randoms_estimate(S, elem1, elem2; n_phi, tau_ns, T_ns=nothing, total=nothing) -> r

Per-event randoms estimate `r_e` for each coincidence, from the singles-per-element
map `S` and the events' two element indices (`elem1`, `elem2` are `(2, N)` rows
`(iz, iphi)`). The shape is `2τ·S_i·S_j`; the absolute scale is set by either the
acquisition time `T_ns` (giving `r = 2τ S_i S_j / T`) or by calibrating the total
to `total` (e.g. the observed number of random coincidences). Returns `Float32`
for use as the reconstruction `contamination`.
"""
function randoms_estimate(S, elem1, elem2; n_phi, tau_ns,
                          T_ns = nothing, total = nothing)
    N = size(elem1, 2)
    base = Vector{Float64}(undef, N)
    @inbounds for e in 1:N
        i = elem_id(elem1[1, e], elem1[2, e], n_phi)
        j = elem_id(elem2[1, e], elem2[2, e], n_phi)
        base[e] = 2 * tau_ns * S[i] * S[j]
    end
    if T_ns !== nothing
        return Float32.(base ./ T_ns)
    elseif total !== nothing
        s = sum(base)
        return Float32.(s > 0 ? base .* (total / s) : base)
    else
        error("randoms_estimate: pass T_ns (absolute scale) or total (calibrated scale)")
    end
end
