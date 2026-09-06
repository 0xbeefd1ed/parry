//! Checks that parry's voxel collision-detection algorithms, which are generic over the
//! [`VoxelQuery`] trait, produce the same results when running on a custom voxel storage
//! as when running on the built-in [`Voxels`] shape.

use parry3d::mass_properties::MassProperties;
use parry3d::math::{IVector, Pose, Real, Vector};
use parry3d::query::details;
use parry3d::query::{
    ContactManifold, DefaultQueryDispatcher, PointQuery, Ray, RayCast, ShapeCastOptions,
};
use parry3d::shape::{Ball, Cuboid, Shape, VoxelData, VoxelQuery, VoxelState, Voxels};
use std::collections::BTreeMap;

/// A custom sparse voxel storage backed by a `BTreeMap`.
///
/// It mirrors the content of a [`Voxels`] shape (including its linear ids) so that query
/// results are directly comparable, but shares none of its implementation.
struct BTreeVoxels {
    voxel_size: Vector,
    domain: [IVector; 2],
    voxels: BTreeMap<[i32; 3], (VoxelState, u32)>,
}

impl BTreeVoxels {
    fn mirroring(voxels: &Voxels) -> Self {
        let mut map = BTreeMap::new();
        for vox in voxels.voxels() {
            if !vox.state.is_empty() {
                map.insert(
                    [vox.grid_coords.x, vox.grid_coords.y, vox.grid_coords.z],
                    (vox.state, vox.linear_id),
                );
            }
        }

        Self {
            voxel_size: voxels.voxel_size(),
            domain: VoxelQuery::domain(voxels),
            voxels: map,
        }
    }
}

impl VoxelQuery for BTreeVoxels {
    fn voxel_size(&self) -> Vector {
        self.voxel_size
    }

    fn domain(&self) -> [IVector; 2] {
        self.domain
    }

    fn voxel_state(&self, key: IVector) -> Option<VoxelState> {
        Some(
            self.voxels
                .get(&[key.x, key.y, key.z])
                .map(|(state, _)| *state)
                .unwrap_or(VoxelState::EMPTY),
        )
    }

    fn linear_id(&self, key: IVector) -> Option<u32> {
        self.voxels.get(&[key.x, key.y, key.z]).map(|(_, id)| *id)
    }

    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = VoxelData> {
        self.voxels.iter().filter_map(move |(k, (state, id))| {
            let key = IVector::new(k[0], k[1], k[2]);
            (key.cmpge(mins).all() && key.cmplt(maxs).all()).then(|| VoxelData {
                linear_id: *id,
                grid_coords: key,
                center: self.voxel_center(key),
                state: *state,
            })
        })
    }
}

/// An 8×8 ground plate, a wall along one of its edges, and a disconnected lone voxel,
/// with non-uniform voxel sizes.
fn reference_shape() -> Voxels {
    let mut keys = vec![];

    for x in 0..8 {
        for z in 0..8 {
            keys.push(IVector::new(x, 0, z));
        }
    }

    for y in 1..4 {
        for z in 0..8 {
            keys.push(IVector::new(0, y, z));
        }
    }

    keys.push(IVector::new(10, 2, 3));

    Voxels::new(Vector::new(1.0, 0.5, 0.75), &keys)
}

fn fixtures() -> (Voxels, BTreeVoxels) {
    let voxels = reference_shape();
    let custom = BTreeVoxels::mirroring(&voxels);
    (voxels, custom)
}

#[test]
fn custom_storage_matches_voxels_states() {
    let (voxels, custom) = fixtures();
    let [mins, maxs] = VoxelQuery::domain(&voxels);
    let margin = IVector::splat(2);

    let mut checked_non_empty = 0;
    for x in mins.x - margin.x..maxs.x + margin.x {
        for y in mins.y - margin.y..maxs.y + margin.y {
            for z in mins.z - margin.z..maxs.z + margin.z {
                let key = IVector::new(x, y, z);
                let state1 = voxels.voxel_state(key).unwrap_or(VoxelState::EMPTY);
                let state2 = VoxelQuery::voxel_state(&custom, key).unwrap_or(VoxelState::EMPTY);
                assert_eq!(state1, state2, "state mismatch at {:?}", key);

                if !state1.is_empty() {
                    assert_eq!(
                        VoxelQuery::linear_id(&voxels, key),
                        VoxelQuery::linear_id(&custom, key),
                        "linear_id mismatch at {:?}",
                        key
                    );
                    checked_non_empty += 1;
                }
            }
        }
    }

    // 8×8 plate + 3×8 wall + 1 lone voxel.
    assert_eq!(checked_non_empty, 64 + 24 + 1);
    assert_eq!(custom.voxels().count(), 64 + 24 + 1);
}

#[test]
fn custom_storage_matches_voxels_mass_properties() {
    let (voxels, custom) = fixtures();
    let density = 2.0;
    let props1 = MassProperties::from_voxels(density, &voxels);
    let props2 = MassProperties::from_voxels(density, &custom);

    // Absolute anchor: 89 voxels of volume 1.0 × 0.5 × 0.75.
    let expected_mass = 89.0 * (1.0 * 0.5 * 0.75) * density;
    assert_relative_eq!(props1.mass(), expected_mass, epsilon = 1.0e-4);

    assert_relative_eq!(props1.mass(), props2.mass(), epsilon = 1.0e-6);
    assert_relative_eq!(props1.local_com, props2.local_com, epsilon = 1.0e-5);
}

#[test]
fn custom_storage_matches_voxels_raycast() {
    let (voxels, custom) = fixtures();

    let mut origins = vec![];
    for i in 0..8 {
        for j in 0..8 {
            // Jittered origins above the shape (jitter avoids exact ties on voxel edges).
            origins.push(Vector::new(
                i as Real * 1.043 + 0.117,
                4.31,
                j as Real * 0.921 + 0.083,
            ));
        }
    }

    let dirs = [
        Vector::new(0.0231, -1.0, 0.0173),
        Vector::new(-0.4173, -0.8317, 0.1531),
        Vector::new(0.723, -0.317, -0.5911),
        Vector::new(0.0731, 1.0, 0.0413), // Away from the shape: must miss.
    ];

    let mut num_hits = 0;
    for origin in &origins {
        for dir in &dirs {
            let ray = Ray::new(*origin, *dir);
            let hit1 = voxels.cast_local_ray_and_get_normal(&ray, 100.0, true);
            let hit2 = details::cast_local_ray_on_voxels(&custom, &ray, 100.0, true);

            assert_eq!(hit1.is_some(), hit2.is_some(), "hit mismatch for {:?}", ray);

            if let (Some(hit1), Some(hit2)) = (hit1, hit2) {
                num_hits += 1;
                assert_relative_eq!(hit1.time_of_impact, hit2.time_of_impact, epsilon = 1.0e-5);
                assert_relative_eq!(hit1.normal, hit2.normal, epsilon = 1.0e-5);
                assert_eq!(hit1.feature, hit2.feature, "feature mismatch for {:?}", ray);
            }
        }
    }

    // Sanity check: the straight-down rays from above the plate must all hit.
    assert!(num_hits >= 64);

    // Absolute anchor: a ray straight above the plate hits its top at y = 0.5.
    let ray = Ray::new(Vector::new(4.13, 4.0, 3.77), Vector::new(0.0, -1.0, 0.0));
    let hit = details::cast_local_ray_on_voxels(&custom, &ray, 100.0, true).unwrap();
    assert_relative_eq!(hit.time_of_impact, 4.0 - 0.5, epsilon = 1.0e-5);
    assert_relative_eq!(hit.normal, Vector::new(0.0, 1.0, 0.0), epsilon = 1.0e-5);
}

#[test]
fn custom_storage_matches_voxels_point_projection() {
    let (voxels, custom) = fixtures();

    let mut points = vec![];
    for i in -2..12 {
        for j in -2..6 {
            for k in -2..10 {
                points.push(Vector::new(
                    i as Real * 1.117 + 0.031,
                    j as Real * 0.617 + 0.043,
                    k as Real * 0.917 + 0.021,
                ));
            }
        }
    }

    for solid in [true, false] {
        for pt in &points {
            let proj1 = voxels.project_local_point(*pt, solid);
            let proj2 = details::project_local_point_on_voxels(&custom, *pt, solid)
                .expect("the shape is not empty")
                .0;

            assert_eq!(
                proj1.is_inside, proj2.is_inside,
                "is_inside mismatch at {:?} (solid: {})",
                pt, solid
            );
            assert_relative_eq!(proj1.point, proj2.point, epsilon = 1.0e-4);
        }
    }
}

type TestManifold = ContactManifold<(), ()>;

fn compare_manifolds(manifolds1: &mut [TestManifold], manifolds2: &mut [TestManifold]) {
    assert_eq!(manifolds1.len(), manifolds2.len());

    // The two backends iterate voxels in a different order, so match manifolds by
    // their subshape ids.
    let sort_key = |m: &TestManifold| (m.subshape1, m.subshape2);
    manifolds1.sort_by_key(sort_key);
    manifolds2.sort_by_key(sort_key);

    for (m1, m2) in manifolds1.iter().zip(manifolds2.iter()) {
        assert_eq!(m1.subshape1, m2.subshape1);
        assert_eq!(m1.subshape2, m2.subshape2);
        assert_eq!(m1.points.len(), m2.points.len());

        if !m1.points.is_empty() {
            assert_relative_eq!(m1.local_n1, m2.local_n1, epsilon = 1.0e-5);
        }

        for (pt1, pt2) in m1.points.iter().zip(m2.points.iter()) {
            assert_relative_eq!(pt1.dist, pt2.dist, epsilon = 1.0e-5);
            assert_relative_eq!(pt1.local_p1, pt2.local_p1, epsilon = 1.0e-4);
            assert_relative_eq!(pt1.local_p2, pt2.local_p2, epsilon = 1.0e-4);
        }
    }
}

#[test]
fn custom_storage_matches_voxels_contact_manifolds() {
    let (voxels, custom) = fixtures();
    let dispatcher = DefaultQueryDispatcher;
    let cuboid = Cuboid::new(Vector::new(0.4, 0.6, 0.5));
    let prediction = 0.05;

    let poses = [
        // Resting on the plate, slightly penetrating.
        Pose::translation(3.13, 0.5 + 0.6 - 0.02, 4.21),
        // Touching both the plate and the wall.
        Pose::translation(1.0 + 0.4 - 0.01, 0.5 + 0.6 - 0.01, 3.87),
        // Hovering within prediction distance.
        Pose::translation(5.11, 0.5 + 0.6 + 0.03, 2.93),
        // Overlapping the lone voxel.
        Pose::translation(10.5, 1.3, 2.71),
        // Far away: no contacts at all.
        Pose::translation(20.0, 10.0, 20.0),
    ];

    let mut total_points = 0;
    for pos12 in &poses {
        let mut manifolds1 = Vec::<TestManifold>::new();
        let mut manifolds2 = Vec::<TestManifold>::new();
        let mut workspace1 = None;
        let mut workspace2 = None;

        details::contact_manifolds_voxels_shape(
            &dispatcher,
            pos12,
            &voxels,
            &cuboid as &dyn Shape,
            prediction,
            &mut manifolds1,
            &mut workspace1,
            false,
        );
        details::contact_manifolds_voxels_shape(
            &dispatcher,
            pos12,
            &custom,
            &cuboid as &dyn Shape,
            prediction,
            &mut manifolds2,
            &mut workspace2,
            false,
        );

        compare_manifolds(&mut manifolds1, &mut manifolds2);
        total_points += manifolds1.iter().map(|m| m.points.len()).sum::<usize>();
    }

    // Sanity check: at least the resting/touching poses must have produced actual contacts.
    assert!(total_points > 0);
}

#[test]
fn custom_storage_matches_voxels_shape_cast() {
    let (voxels, custom) = fixtures();
    let dispatcher = DefaultQueryDispatcher;
    let ball = Ball::new(0.3);
    let pos12 = Pose::translation(4.05, 3.0, 4.1);
    let vel12 = Vector::new(0.0, -1.0, 0.0);
    let options = ShapeCastOptions::default();

    let hit1 =
        details::cast_shapes_voxels_shape(&dispatcher, &pos12, vel12, &voxels, &ball, options);
    let hit2 =
        details::cast_shapes_voxels_shape(&dispatcher, &pos12, vel12, &custom, &ball, options);

    let hit1 = hit1.expect("the ball must hit the plate");
    let hit2 = hit2.expect("the ball must hit the plate");

    // Absolute anchor: the ball surface reaches the plate's top (y = 0.5) after
    // travelling 3.0 - 0.5 - 0.3 units.
    assert_relative_eq!(hit1.time_of_impact, 3.0 - 0.5 - 0.3, epsilon = 1.0e-4);

    assert_relative_eq!(hit1.time_of_impact, hit2.time_of_impact, epsilon = 1.0e-5);
    assert_relative_eq!(hit1.normal1, hit2.normal1, epsilon = 1.0e-4);
    assert_relative_eq!(hit1.witness1, hit2.witness1, epsilon = 1.0e-4);
}

#[test]
fn custom_storage_matches_voxels_intersection_test() {
    let (voxels, custom) = fixtures();
    let dispatcher = DefaultQueryDispatcher;
    let cuboid = Cuboid::new(Vector::new(0.4, 0.6, 0.5));

    let poses = [
        (Pose::translation(3.13, 0.7, 4.21), true), // Overlapping the plate.
        (Pose::translation(10.5, 1.3, 2.71), true), // Overlapping the lone voxel.
        (Pose::translation(4.0, 5.0, 4.0), false),  // Above everything.
        (Pose::translation(20.0, 0.0, 20.0), false), // Far away.
    ];

    for (pos12, expected) in &poses {
        let hit1 = details::intersection_test_voxels_shape(&dispatcher, pos12, &voxels, &cuboid);
        let hit2 = details::intersection_test_voxels_shape(&dispatcher, pos12, &custom, &cuboid);
        assert_eq!(hit1, *expected);
        assert_eq!(hit2, *expected);
    }
}
