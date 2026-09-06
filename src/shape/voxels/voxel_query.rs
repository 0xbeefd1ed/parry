use crate::math::{ivect_to_vect, vect_to_ivect, IVector, Vector};

use crate::bounding_volume::Aabb;
use crate::shape::{VoxelData, VoxelState, Voxels};

/// Abstraction over the storage of a shape made of axis-aligned, uniformly sized voxels.
///
/// Parry's voxel collision-detection algorithms (contact manifolds, intersection tests,
/// linear and nonlinear shape-casting, ray-casting, point projection, mass properties) are
/// written against this trait rather than against the concrete [`Voxels`] shape. Implementing
/// it for a custom sparse data-structure (chunked grid, octree, VDB-like tree, etc.) lets these
/// algorithms run directly on that structure without copying it into a [`Voxels`] shape,
/// typically by calling the generic query functions from a custom
/// [`QueryDispatcher`](crate::query::QueryDispatcher).
///
/// # Grid conventions
///
/// Voxels are identified by their integer grid coordinates `key`. The voxel with coordinates
/// `key` covers the world-space (well, shape-local-space) range
/// `[key * voxel_size, (key + 1) * voxel_size]`, so its center is at
/// `(key + 0.5) * voxel_size`. Grid ranges are always given as semi-open intervals
/// `[mins, maxs)`: `mins` is included, `maxs` is excluded.
///
/// # Neighborhood states
///
/// Each non-empty voxel must know which of its immediate axis-aligned neighbors are also
/// non-empty, exposed as a [`VoxelState`]. This is what allows collision-detection to avoid
/// hitting the "internal edges" between adjacent voxels. Implementors can either store this
/// information (like [`Voxels`] does, one byte per voxel), or derive it on the fly from
/// occupancy data using [`VoxelState::with_filled_neighbors`].
///
/// # Note for implementors
///
/// This trait is not dyn-compatible (`voxels_in_range` returns `impl Iterator`). The generic
/// query functions are monomorphized for each storage type. To plug a custom storage into a
/// physics pipeline, wrap it in a type implementing [`Shape`](crate::shape::Shape) (typically
/// with [`ShapeType::Custom`](crate::shape::ShapeType::Custom)) and dispatch to the generic
/// voxel query functions from a custom `QueryDispatcher`.
///
/// # Example
///
/// Implementing `VoxelQuery` for a dense boolean grid, then running one of parry's generic
/// algorithms on it:
///
/// ```
/// # #[cfg(all(feature = "dim3", feature = "f32"))] {
/// use parry3d::mass_properties::MassProperties;
/// use parry3d::math::{IVector, Vector};
/// use parry3d::shape::{AxisMask, VoxelData, VoxelQuery, VoxelState};
///
/// const N: i32 = 4;
///
/// /// A dense 4×4×4 grid of voxels, each of size 1×1×1.
/// struct DenseGrid {
///     cells: [[[bool; 4]; 4]; 4],
/// }
///
/// impl DenseGrid {
///     fn filled(&self, key: IVector) -> bool {
///         key.cmpge(IVector::ZERO).all()
///             && key.cmplt(IVector::splat(N)).all()
///             && self.cells[key.x as usize][key.y as usize][key.z as usize]
///     }
/// }
///
/// impl VoxelQuery for DenseGrid {
///     fn voxel_size(&self) -> Vector {
///         Vector::splat(1.0)
///     }
///
///     fn domain(&self) -> [IVector; 2] {
///         [IVector::ZERO, IVector::splat(N)]
///     }
///
///     fn voxel_state(&self, key: IVector) -> Option<VoxelState> {
///         if !self.filled(key) {
///             return Some(VoxelState::EMPTY);
///         }
///
///         let mut mask = AxisMask::empty();
///         if self.filled(key + IVector::new(1, 0, 0)) { mask |= AxisMask::X_POS; }
///         if self.filled(key - IVector::new(1, 0, 0)) { mask |= AxisMask::X_NEG; }
///         if self.filled(key + IVector::new(0, 1, 0)) { mask |= AxisMask::Y_POS; }
///         if self.filled(key - IVector::new(0, 1, 0)) { mask |= AxisMask::Y_NEG; }
///         if self.filled(key + IVector::new(0, 0, 1)) { mask |= AxisMask::Z_POS; }
///         if self.filled(key - IVector::new(0, 0, 1)) { mask |= AxisMask::Z_NEG; }
///         Some(VoxelState::with_filled_neighbors(mask))
///     }
///
///     fn linear_id(&self, key: IVector) -> Option<u32> {
///         self.filled(key)
///             .then(|| (key.x * N * N + key.y * N + key.z) as u32)
///     }
///
///     fn voxels_in_range(
///         &self,
///         mins: IVector,
///         maxs: IVector,
///     ) -> impl Iterator<Item = VoxelData> {
///         let mins = mins.max(IVector::ZERO);
///         let maxs = maxs.min(IVector::splat(N));
///         (mins.x..maxs.x).flat_map(move |x| {
///             (mins.y..maxs.y).flat_map(move |y| {
///                 (mins.z..maxs.z).filter_map(move |z| {
///                     let key = IVector::new(x, y, z);
///                     let state = self.voxel_state(key)?;
///                     (!state.is_empty()).then(|| VoxelData {
///                         linear_id: self.linear_id(key).unwrap(),
///                         grid_coords: key,
///                         center: self.voxel_center(key),
///                         state,
///                     })
///                 })
///             })
///         })
///     }
/// }
///
/// let mut grid = DenseGrid { cells: [[[true; 4]; 4]; 4] };
/// // Any of parry's generic voxel algorithms now runs on `DenseGrid` directly:
/// let props = MassProperties::from_voxels(1.0, &grid);
/// assert_eq!(props.mass(), 64.0);
/// # }
/// ```
pub trait VoxelQuery {
    /// The size of each voxel along each local coordinate axis.
    fn voxel_size(&self) -> Vector;

    /// The semi-open range `[mins, maxs)` of grid coordinates covered by this shape.
    ///
    /// This must be a conservative bound: every non-empty voxel must lie within the returned
    /// range, but the range may also cover empty voxels.
    fn domain(&self) -> [IVector; 2];

    /// The state of the voxel at the given grid coordinates.
    ///
    /// Both `None` and `Some(VoxelState::EMPTY)` designate an empty voxel; by convention,
    /// `None` is returned when `key` falls outside of the storage's tracked domain.
    fn voxel_state(&self, key: IVector) -> Option<VoxelState>;

    /// A stable identifier of the voxel at the given grid coordinates.
    ///
    /// The identifier must be unique among the currently stored voxels and must match the
    /// value of [`VoxelData::linear_id`] yielded by [`Self::voxels_in_range`] for the same
    /// voxel. It is used to build [`FeatureId`](crate::shape::FeatureId)s in query results
    /// and to match contact points across frames, so it should remain stable as long as the
    /// shape isn't modified. Returns `None` if no identifier is associated to this
    /// coordinate (e.g. empty voxel in unallocated storage).
    fn linear_id(&self, key: IVector) -> Option<u32>;

    /// Iterates through the voxels within the given semi-open grid coordinate range.
    ///
    /// Implementations must yield every non-empty voxel with grid coordinates in
    /// `[mins, maxs)` exactly once. They may additionally yield empty voxels within that
    /// range (callers filter on [`VoxelData::state`]), but must never yield a voxel outside
    /// of the range.
    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = VoxelData>;

    /// Iterates through every voxel of this shape.
    ///
    /// This is equivalent to [`Self::voxels_in_range`] applied to the whole [`Self::domain`].
    fn voxels(&self) -> impl Iterator<Item = VoxelData> {
        let [mins, maxs] = self.domain();
        self.voxels_in_range(mins, maxs)
    }

    /// Iterates through every voxel intersecting the given local-space AABB.
    fn voxels_intersecting_local_aabb(&self, aabb: &Aabb) -> impl Iterator<Item = VoxelData> {
        let [mins, maxs] = self.voxel_range_intersecting_local_aabb(aabb);
        self.voxels_in_range(mins, maxs)
    }

    /// The grid coordinates of the voxel containing the given local-space point.
    ///
    /// The returned coordinates are valid regardless of whether the corresponding voxel
    /// is filled, empty, or outside of [`Self::domain`].
    fn voxel_at_point(&self, point: Vector) -> IVector {
        vect_to_ivect((point / self.voxel_size()).floor())
    }

    /// The local-space center of the voxel with the given grid coordinates.
    fn voxel_center(&self, key: IVector) -> Vector {
        (ivect_to_vect(key) + Vector::splat(0.5)) * self.voxel_size()
    }

    /// The local-space AABB of the voxel with the given grid coordinates.
    fn voxel_aabb(&self, key: IVector) -> Aabb {
        let center = self.voxel_center(key);
        Aabb::from_half_extents(center, self.voxel_size() / 2.0)
    }

    /// The semi-open range of grid coordinates of the voxels intersecting the given AABB.
    ///
    /// The returned range covers both empty and non-empty voxels, and is not limited to the
    /// bounds defined by [`Self::domain`].
    fn voxel_range_intersecting_local_aabb(&self, aabb: &Aabb) -> [IVector; 2] {
        let mins = vect_to_ivect((aabb.mins / self.voxel_size()).floor());
        let maxs = vect_to_ivect((aabb.maxs / self.voxel_size()).ceil());
        [mins, maxs]
    }

    /// The local-space AABB of the given semi-open range of voxel grid coordinates.
    fn voxel_range_aabb(&self, mins: IVector, maxs: IVector) -> Aabb {
        Aabb {
            mins: ivect_to_vect(mins) * self.voxel_size(),
            maxs: ivect_to_vect(maxs) * self.voxel_size(),
        }
    }

    /// Aligns the given AABB with the voxelized grid.
    ///
    /// The returned AABB has corners lying at the grid intersections (i.e. matches voxel
    /// corners) and fully contains the input `aabb`.
    fn align_aabb_to_grid(&self, aabb: &Aabb) -> Aabb {
        let mins = (aabb.mins / self.voxel_size()).floor() * self.voxel_size();
        let maxs = (aabb.maxs / self.voxel_size()).ceil() * self.voxel_size();
        Aabb { mins, maxs }
    }

    /// The local-space AABB of this voxels shape.
    fn local_aabb(&self) -> Aabb {
        let [mins, maxs] = self.domain();
        self.voxel_range_aabb(mins, maxs)
    }
}

impl VoxelQuery for Voxels {
    #[inline]
    fn voxel_size(&self) -> Vector {
        self.voxel_size()
    }

    #[inline]
    fn domain(&self) -> [IVector; 2] {
        self.domain()
    }

    #[inline]
    fn voxel_state(&self, key: IVector) -> Option<VoxelState> {
        self.voxel_state(key)
    }

    #[inline]
    fn linear_id(&self, key: IVector) -> Option<u32> {
        self.linear_index(key).map(|id| id.flat_id() as u32)
    }

    #[inline]
    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = VoxelData> {
        self.voxels_in_range(mins, maxs)
    }

    #[inline]
    fn voxels(&self) -> impl Iterator<Item = VoxelData> {
        self.voxels()
    }

    #[inline]
    fn local_aabb(&self) -> Aabb {
        self.local_aabb()
    }
}
