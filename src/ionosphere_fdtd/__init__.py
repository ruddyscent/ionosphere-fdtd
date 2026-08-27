"""Geodesic finite-difference time-domain Earth-ionosphere model."""

from .adaptive_mesh import (
    AdaptiveMeshValidation,
    SphericalRefinementRegion,
    build_adaptive_geodesic_mesh,
    validate_adaptive_mesh,
)
from ._torch_runtime import BackendUnavailableError
from .checkpoint import CheckpointError
from .data_artifacts import (
    DataArtifactError,
    DatasetProvenance,
    MeshMaterialArtifact,
    VariableProvenance,
)
from .distributed import (
    DistributedGeodesicFDTD,
    TorchDistributedHaloExchange,
    initialize_torchrun_process_group,
)
from .materials import (
    EarthIonosphereMaterial,
    GriddedMaterial,
    LayeredEarthIonosphereMaterial,
    MaterialUpdateCoefficientTensors,
    SampledMaterialTensors,
    SphericalAnomaly,
    SpatialEarthIonosphereMaterial,
)
from .mesh import (
    GeodesicMesh,
    build_geodesic_mesh,
    build_geodesic_mesh_from_topology,
)
from .partition import (
    FieldHalo,
    PartitionValidation,
    RankSurfacePartition,
    SurfacePartition,
    partition_surface_mesh,
    validate_surface_partition,
)
from .plasma import (
    ColdPlasmaSpecies,
    MeshPlasmaModel,
    PlasmaCoefficientTensors,
    PlasmaSpeciesCoefficientTensors,
)
from .radial_grid import (
    RadialGridValidation,
    RadialRefinementRegion,
    build_refined_radial_grid,
    validate_radial_grid,
)
from .solver import GeodesicFDTD, SimulationConfig
from .sources import GaussianCurrent, TangentialGaussianCurrent
from .surface_impedance import (
    ConductiveHalfSpaceSurface,
    SurfaceImpedanceCoefficientTensors,
)
from .visualization import (
    RadialSection,
    Receiver,
    ReceiverTraces,
    animate_surface_field,
    plot_mesh_3d,
    plot_radial_section,
    plot_receiver_traces,
    plot_surface_field,
    record_receiver_traces,
    run_live_surface,
    sample_radial_section,
)

__all__ = [
    "AdaptiveMeshValidation",
    "BackendUnavailableError",
    "CheckpointError",
    "ColdPlasmaSpecies",
    "ConductiveHalfSpaceSurface",
    "DataArtifactError",
    "DatasetProvenance",
    "DistributedGeodesicFDTD",
    "EarthIonosphereMaterial",
    "FieldHalo",
    "GriddedMaterial",
    "LayeredEarthIonosphereMaterial",
    "MaterialUpdateCoefficientTensors",
    "MeshMaterialArtifact",
    "MeshPlasmaModel",
    "GaussianCurrent",
    "TangentialGaussianCurrent",
    "GeodesicFDTD",
    "GeodesicMesh",
    "PartitionValidation",
    "PlasmaCoefficientTensors",
    "PlasmaSpeciesCoefficientTensors",
    "RadialSection",
    "RadialGridValidation",
    "RadialRefinementRegion",
    "Receiver",
    "ReceiverTraces",
    "RankSurfacePartition",
    "SimulationConfig",
    "SampledMaterialTensors",
    "SphericalRefinementRegion",
    "SphericalAnomaly",
    "SurfacePartition",
    "SpatialEarthIonosphereMaterial",
    "SurfaceImpedanceCoefficientTensors",
    "TorchDistributedHaloExchange",
    "VariableProvenance",
    "animate_surface_field",
    "build_geodesic_mesh",
    "build_geodesic_mesh_from_topology",
    "build_adaptive_geodesic_mesh",
    "build_refined_radial_grid",
    "initialize_torchrun_process_group",
    "plot_mesh_3d",
    "partition_surface_mesh",
    "plot_radial_section",
    "plot_receiver_traces",
    "plot_surface_field",
    "record_receiver_traces",
    "run_live_surface",
    "sample_radial_section",
    "validate_adaptive_mesh",
    "validate_radial_grid",
    "validate_surface_partition",
]
