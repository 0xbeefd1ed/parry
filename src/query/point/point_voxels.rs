use crate::bounding_volume::{Aabb, BoundingVolume};
use crate::math::{Real, Vector};
use crate::query::{PointProjection, PointQuery};
use crate::shape::{Cuboid, FeatureId, VoxelQuery, Voxels, VoxelsChunkRef};

/// Projects a point on a voxel shape represented by any storage implementing [`VoxelQuery`].
///
/// Returns the projection and the [`linear_id`](VoxelQuery::linear_id) of the voxel the
/// projected point lies on, or `None` if the shape contains no voxel. If `solid` is `true` and
/// the point lies inside a non-empty voxel, the point itself is returned as the projection.
///
/// This searches voxels in growing regions centered on the point, without relying on any
/// acceleration structure. The concrete [`Voxels`] shape implements [`PointQuery`] with a
/// faster search based on its internal BVH; this function is mostly useful for implementing
/// point queries on custom voxel storages.
///
/// Like the [`PointQuery`] implementation of [`Voxels`], the non-solid projection of a point
/// lying inside of the shape is approximated: the point is projected on the boundary of the
/// closest voxel, which isn't necessarily on the boundary of the union of all the voxels.
pub fn project_local_point_on_voxels<V: ?Sized + VoxelQuery>(
    voxels: &V,
    pt: Vector,
    solid: bool,
) -> Option<(PointProjection, u32)> {
    let base_cuboid = Cuboid::new(voxels.voxel_size() / 2.0);

    // Fast path: the point lies inside a non-empty voxel.
    let key_at_pt = voxels.voxel_at_point(pt);
    if solid
        && voxels
            .voxel_state(key_at_pt)
            .is_some_and(|state| !state.is_empty())
    {
        return Some((
            PointProjection::new(true, pt),
            voxels.linear_id(key_at_pt).unwrap_or(0),
        ));
    }

    // The distance from `pt` to the domain’s AABB is a lower-bound of the distance from `pt`
    // to its projection on the shape. Use it to initialize the search radius.
    let domain_aabb = voxels.local_aabb();
    let mut search_radius =
        domain_aabb.distance_to_local_point(pt, true) + voxels.voxel_size().max_element();

    loop {
        let search_aabb = Aabb::from_half_extents(pt, Vector::splat(search_radius));
        let mut best: Option<(PointProjection, u32)> = None;
        let mut best_dist = Real::MAX;

        for vox in voxels.voxels_intersecting_local_aabb(&search_aabb) {
            if vox.state.is_empty() {
                continue;
            }

            let mut candidate = base_cuboid.project_local_point(pt - vox.center, solid);
            candidate.point += vox.center;

            let candidate_dist = (candidate.point - pt).length();
            if candidate_dist < best_dist {
                best = Some((candidate, vox.linear_id));
                best_dist = candidate_dist;
            }
        }

        if let Some(best) = best {
            if best_dist <= search_radius {
                // Any voxel closer to `pt` than `best_dist` would have intersected the
                // search region, so this projection is optimal.
                return Some(best);
            }

            // The projection found lies outside of the search region: another voxel closer
            // to its boundary could still be a better candidate. Re-run with a search region
            // that covers every possibly-better voxel.
            search_radius = best_dist;
        } else {
            if search_aabb.contains(&domain_aabb) {
                // The whole shape was searched and no voxel was found.
                return None;
            }

            search_radius *= 2.0;
        }
    }
}

impl PointQuery for Voxels {
    #[inline]
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        self.chunk_bvh()
            .project_point(pt, Real::MAX, |chunk_id, _| {
                let chunk = self.chunk_ref(chunk_id);
                chunk
                    .project_local_point_and_get_vox_id(pt, solid)
                    .map(|(proj, _)| proj)
            })
            .map(|res| res.1 .1)
            .unwrap_or(PointProjection::new(false, Vector::splat(Real::MAX)))
    }

    #[inline]
    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        self.chunk_bvh()
            .project_point_and_get_feature(pt, Real::MAX, |chunk_id, _| {
                let chunk = self.chunk_ref(chunk_id);
                // TODO: we need a way to return both the voxel id, and the feature on the voxel.
                chunk
                    .project_local_point_and_get_vox_id(pt, false)
                    .map(|(proj, vox)| (proj, FeatureId::Face(vox)))
            })
            .map(|res| res.1 .1)
            .unwrap_or((
                PointProjection::new(false, Vector::splat(Real::MAX)),
                FeatureId::Unknown,
            ))
    }
}

impl<'a> VoxelsChunkRef<'a> {
    #[inline]
    fn project_local_point_and_get_vox_id(
        &self,
        pt: Vector,
        solid: bool,
    ) -> Option<(PointProjection, u32)> {
        // TODO: optimize this naive implementation that just iterates on all the voxels
        //       from this chunk.
        let base_cuboid = Cuboid::new(self.parent.voxel_size() / 2.0);
        let mut smallest_dist = Real::MAX;
        let mut result = PointProjection::new(false, pt);
        let mut result_vox_id = 0;

        for vox in self.voxels() {
            let mut candidate = base_cuboid.project_local_point(pt - vox.center, solid);
            candidate.point += vox.center;

            let candidate_dist = (candidate.point - pt).length();
            if candidate_dist < smallest_dist {
                result = candidate;
                result_vox_id = vox.linear_id;
                smallest_dist = candidate_dist;
            }
        }

        (smallest_dist < Real::MAX).then_some((result, result_vox_id))
    }
}
