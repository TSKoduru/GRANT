import numpy as np
import open3d as o3d

try:
    from .scan_types import CapturedView, PointCloud, Pose6D
except ImportError:
    from scan_types import CapturedView, PointCloud, Pose6D


class Registration:
    """
    Point-cloud fusion + mesh reconstruction.

    Two reconstruction paths:
      - `reconstruct_mesh(cloud)`    → Poisson on a merged point cloud.
      - `tsdf_fuse(views, …)`        → TSDF integration over RGBD frames.
        TSDF is preferred for the fixed-algorithm pipeline because it
        handles per-view weighting and produces cleaner meshes when
        you have accurate per-frame depth + pose.

    Alignment primitives:
      - `merge(existing, new, run_icp)`    — local ICP refinement.
      - `global_align(source, target)`     — RANSAC (FPFH) + ICP. Use this
        between the two object orientations, where ICP alone can't converge
        because the relative rotation is ~180°.
    """

    # Local registration / fusion
    VOXEL_SIZE = 0.002                 # 2mm; tighter = more detail, more memory
    ICP_MAX_DIST = 0.01                # 1cm correspondence threshold
    ICP_MAX_ITER = 50
    NORMAL_RADIUS = 0.01
    NORMAL_KNN = 30

    # Global (cross-orientation) registration
    GLOBAL_VOXEL_SIZE = 0.005          # 5mm; coarse enough for fast FPFH
    GLOBAL_NORMAL_RADIUS = 0.01
    GLOBAL_FEATURE_RADIUS = 0.025
    GLOBAL_FEATURE_KNN = 100
    RANSAC_MAX_CORR_DIST = 0.0075      # 1.5 * GLOBAL_VOXEL_SIZE
    RANSAC_N = 4                       # correspondences sampled per iteration
    RANSAC_MAX_ITER = 100_000
    RANSAC_CONFIDENCE = 0.999

    # Poisson (legacy path)
    POISSON_DEPTH = 9
    POISSON_DENSITY_QUANTILE = 0.05

    # TSDF
    TSDF_VOXEL = 0.002                 # 2mm voxels
    TSDF_TRUNC = 0.008                 # 4 * voxel_size is a solid default
    TSDF_DEPTH_TRUNC = 0.5             # discard depth readings past 0.5m

    # ──────────────────────────────────────────────────────────────────
    # Per-view transforms
    # ──────────────────────────────────────────────────────────────────

    def transform_to_world(
        self,
        cloud: PointCloud,
        camera_pose: Pose6D,
    ) -> PointCloud:
        if cloud.points.shape[0] == 0:
            return PointCloud(
                points=cloud.points.copy(),
                colors=None if cloud.colors is None else cloud.colors.copy(),
            )
        T = _pose_to_matrix(camera_pose)
        pts_h = np.hstack([cloud.points, np.ones((cloud.points.shape[0], 1))])
        world_pts = (pts_h @ T.T)[:, :3]
        return PointCloud(
            points=world_pts.astype(np.float32),
            colors=None if cloud.colors is None else cloud.colors.copy(),
        )

    def merge(
        self,
        existing: PointCloud,
        new_cloud: PointCloud,
        run_icp: bool = True,
    ) -> PointCloud:
        if new_cloud.points.shape[0] == 0:
            return existing
        if existing.points.shape[0] == 0:
            merged = _to_o3d(new_cloud).voxel_down_sample(self.VOXEL_SIZE)
            return _from_o3d(merged)

        target = _to_o3d(existing)
        source = _to_o3d(new_cloud)

        if run_icp:
            # Point-to-point: more stable than point-to-plane on low-curvature
            # surfaces where plane-normal gradients are ambiguous.
            result = o3d.pipelines.registration.registration_icp(
                source=source,
                target=target,
                max_correspondence_distance=self.ICP_MAX_DIST,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.ICP_MAX_ITER
                ),
            )
            source.transform(result.transformation)

        merged = (target + source).voxel_down_sample(self.VOXEL_SIZE)
        return _from_o3d(merged)

    # ──────────────────────────────────────────────────────────────────
    # Cross-orientation alignment (RANSAC + ICP)
    # ──────────────────────────────────────────────────────────────────

    def global_align(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> tuple[np.ndarray, float]:
        """
        Align `source` onto `target` when the initial relative pose is unknown
        (post-flip case). Two-stage: RANSAC on FPFH features for coarse
        alignment, then ICP for refinement.

        Returns (T_source_to_target 4x4, fitness ∈ [0,1]).
        Fitness is the ICP overlap ratio — anything below ~0.4 means the
        flip left almost no shared surface and the alignment is suspect.
        """
        if source.points.shape[0] < 100 or target.points.shape[0] < 100:
            raise ValueError("global_align needs ≥100 points per cloud")

        src_down, src_fpfh = _preprocess_for_ransac(
            source,
            self.GLOBAL_VOXEL_SIZE,
            self.GLOBAL_NORMAL_RADIUS,
            self.GLOBAL_FEATURE_RADIUS,
            self.GLOBAL_FEATURE_KNN,
        )
        tgt_down, tgt_fpfh = _preprocess_for_ransac(
            target,
            self.GLOBAL_VOXEL_SIZE,
            self.GLOBAL_NORMAL_RADIUS,
            self.GLOBAL_FEATURE_RADIUS,
            self.GLOBAL_FEATURE_KNN,
        )

        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=src_down,
            target=tgt_down,
            source_feature=src_fpfh,
            target_feature=tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.RANSAC_MAX_CORR_DIST,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=self.RANSAC_N,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.RANSAC_MAX_CORR_DIST
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.RANSAC_MAX_ITER, self.RANSAC_CONFIDENCE
            ),
        )

        # ICP refinement on the full-resolution clouds
        src_full = _to_o3d(source)
        tgt_full = _to_o3d(target)
        icp = o3d.pipelines.registration.registration_icp(
            source=src_full,
            target=tgt_full,
            max_correspondence_distance=self.ICP_MAX_DIST,
            init=ransac.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.ICP_MAX_ITER
            ),
        )
        return icp.transformation, float(icp.fitness)

    def apply_transform_to_views(
        self,
        views: list[CapturedView],
        T: np.ndarray,
    ) -> list[CapturedView]:
        """Pre-multiply each view's camera-to-world pose by T."""
        out: list[CapturedView] = []
        for v in views:
            new_T = T @ _pose_to_matrix(v.camera_pose)
            out.append(CapturedView(frame=v.frame, camera_pose=_matrix_to_pose(new_T)))
        return out

    # ──────────────────────────────────────────────────────────────────
    # TSDF fusion
    # ──────────────────────────────────────────────────────────────────

    def tsdf_fuse(self, views: list[CapturedView]) -> o3d.geometry.TriangleMesh:
        """
        Integrate every view's depth into a scalable TSDF volume and
        extract the zero-level-set mesh.

        Each view's `camera_pose` is the camera-to-world transform; TSDF
        needs the inverse (world-to-camera) as its extrinsic.
        """
        if not views:
            raise ValueError("tsdf_fuse needs at least one view")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.TSDF_VOXEL,
            sdf_trunc=self.TSDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for v in views:
            intr = v.frame.intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
            )

            depth = v.frame.depth.astype(np.float32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(v.frame.rgb)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,           # depth already in meters
                depth_trunc=self.TSDF_DEPTH_TRUNC,
                convert_rgb_to_intensity=False,
            )

            extrinsic = np.linalg.inv(_pose_to_matrix(v.camera_pose))
            volume.integrate(rgbd, intrinsic, extrinsic)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh

    # ──────────────────────────────────────────────────────────────────
    # Poisson (legacy — kept for point-cloud-only callers)
    # ──────────────────────────────────────────────────────────────────

    def reconstruct_mesh(self, cloud: PointCloud) -> o3d.geometry.TriangleMesh:
        if cloud.points.shape[0] < 100:
            raise ValueError(
                f"reconstruct_mesh needs ≥100 points, got {cloud.points.shape[0]}"
            )
        pcd = _to_o3d(cloud)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.NORMAL_RADIUS * 3, max_nn=self.NORMAL_KNN
            )
        )
        centroid = pcd.get_center()
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        outward = points - centroid
        flip = np.einsum("ij,ij->i", normals, outward) < 0
        normals[flip] *= -1.0
        pcd.normals = o3d.utility.Vector3dVector(normals)

        mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.POISSON_DEPTH
            )
        )
        densities = np.asarray(densities)
        threshold = np.quantile(densities, self.POISSON_DENSITY_QUANTILE)
        mesh.remove_vertices_by_mask(densities < threshold)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _preprocess_for_ransac(
    cloud: PointCloud,
    voxel: float,
    normal_radius: float,
    feature_radius: float,
    feature_knn: int,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd = _to_o3d(cloud).voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius, max_nn=feature_knn
        ),
    )
    return pcd, fpfh


def _pose_to_matrix(pose: Pose6D) -> np.ndarray:
    q = np.array([pose.qx, pose.qy, pose.qz, pose.qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        R = np.eye(3)
    else:
        qx, qy, qz, qw = q / n
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
        ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pose.x, pose.y, pose.z]
    return T


def _matrix_to_pose(T: np.ndarray) -> Pose6D:
    R = T[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return Pose6D(
        x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3]),
        qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw),
    )


def _to_o3d(cloud: PointCloud) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))
    if cloud.colors is not None and cloud.colors.shape[0] == cloud.points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(
            cloud.colors.astype(np.float64) / 255.0
        )
    return pcd


def _from_o3d(pcd: o3d.geometry.PointCloud) -> PointCloud:
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    return PointCloud(points=points, colors=colors)
