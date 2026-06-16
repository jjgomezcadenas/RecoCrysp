# API reference

## Module

```@docs
RecoCrysp
```

## Projectors

```@docs
joseph3d_fwd
joseph3d_fwd!
joseph3d_back
joseph3d_back!
```

## PET scanner geometry

```@docs
RegularPolygonPETScannerGeometry
RegularPolygonPETLORDescriptor
get_lor_endpoints
get_lor_coordinates
```

## Reconstruction

```@docs
sensitivity_image
ListmodePoissonModel
predicted
neg_log_likelihood
em_update
mlem
osem
subset_models
```

## Internals

```@docs
RecoCrysp.ray_cube_intersection_joseph
```
