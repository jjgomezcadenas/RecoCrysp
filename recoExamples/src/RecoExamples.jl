"""
    RecoExamples

Support code for the RecoCrysp Monte-Carlo reconstruction studies (Part III):
readers and estimators for PTCRYSP simulated data, kept out of the lean
`RecoCrysp` core (which depends only on KernelAbstractions). The per-study run
scripts live under `recoExamples/<phantom>/` and use these helpers.
"""
module RecoExamples

include("mc_listmode.jl")
include("randoms.jl")
include("norm_lors.jl")
include("attenuation.jl")
include("background.jl")
include("nema_phantom.jl")

export MCCoincidences, read_coincidences, endpoints,
       is_true, is_scatter, is_random, is_single_scatter, is_multiple_scatter
export elem_id, singles_element_counts, randoms_estimate
export emission_sens_lors, surface_doi_lors, ideal_sphere_lors
export sphere_chord, attenuation_factors
export lor_sinogram_coords, background_estimate, background_sinograms, gaussian_postfilter
export NemaSphere, NEMA_SPHERES, NEMA_HOT_RATIO, NEMA_BODY_R_MM,
       nema_sphere_masks, nema_background_mask, nema_true_image

end # module
