#!/usr/bin/env python3
"""
Axis-aware FBX skinning-driven Panda3D hand viewer.
Uses hierarchical swing-twist style quaternion driving without offset JSON.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Filename,
    Material,
    NodePath,
    Point3,
    Quat,
    Vec3,
    Vec4,
    WindowProperties,
)


def _to_model_path(p: Path) -> str:
    return Filename.fromOsSpecific(str(p.resolve())).getFullpath()


def _finger(prefix: str, name: str) -> List[str]:
    if name == "Thumb":
        return [
            f"{prefix}_{name}Metacarpal",
            f"{prefix}_{name}Proximal",
            f"{prefix}_{name}Distal",
        ]
    return [
        f"{prefix}_{name}Metacarpal",
        f"{prefix}_{name}Proximal",
        f"{prefix}_{name}Intermediate",
        f"{prefix}_{name}Distal",
    ]


LANDMARK_TO_BONE_SEQ_LEFT = {
    "wrist": (0, "L_Wrist", 9),
    "thumb": [
        (1, _finger("L", "Thumb")[0], 2),
        (2, _finger("L", "Thumb")[1], 3),
        (3, _finger("L", "Thumb")[2], 4),
    ],
    "index": [
        (0, _finger("L", "Index")[0], 5),
        (5, _finger("L", "Index")[1], 6),
        (6, _finger("L", "Index")[2], 7),
        (7, _finger("L", "Index")[3], 8),
    ],
    "middle": [
        (0, _finger("L", "Middle")[0], 9),
        (9, _finger("L", "Middle")[1], 10),
        (10, _finger("L", "Middle")[2], 11),
        (11, _finger("L", "Middle")[3], 12),
    ],
    "ring": [
        (0, _finger("L", "Ring")[0], 13),
        (13, _finger("L", "Ring")[1], 14),
        (14, _finger("L", "Ring")[2], 15),
        (15, _finger("L", "Ring")[3], 16),
    ],
    "little": [
        (0, _finger("L", "Little")[0], 17),
        (17, _finger("L", "Little")[1], 18),
        (18, _finger("L", "Little")[2], 19),
        (19, _finger("L", "Little")[3], 20),
    ],
}

LANDMARK_TO_BONE_SEQ_RIGHT = {
    "wrist": (0, "R_Wrist", 9),
    "thumb": [
        (1, _finger("R", "Thumb")[0], 2),
        (2, _finger("R", "Thumb")[1], 3),
        (3, _finger("R", "Thumb")[2], 4),
    ],
    "index": [
        (0, _finger("R", "Index")[0], 5),
        (5, _finger("R", "Index")[1], 6),
        (6, _finger("R", "Index")[2], 7),
        (7, _finger("R", "Index")[3], 8),
    ],
    "middle": [
        (0, _finger("R", "Middle")[0], 9),
        (9, _finger("R", "Middle")[1], 10),
        (10, _finger("R", "Middle")[2], 11),
        (11, _finger("R", "Middle")[3], 12),
    ],
    "ring": [
        (0, _finger("R", "Ring")[0], 13),
        (13, _finger("R", "Ring")[1], 14),
        (14, _finger("R", "Ring")[2], 15),
        (15, _finger("R", "Ring")[3], 16),
    ],
    "little": [
        (0, _finger("R", "Little")[0], 17),
        (17, _finger("R", "Little")[1], 18),
        (18, _finger("R", "Little")[2], 19),
        (19, _finger("R", "Little")[3], 20),
    ],
}

_PX = Vec3(1, 0, 0)
_NX = Vec3(-1, 0, 0)
_PY = Vec3(0, 1, 0)
_PZ = Vec3(0, 0, 1)
_NZ = Vec3(0, 0, -1)

# From measured rest-pose probe results.
LEFT_FORWARD_AXIS: Dict[str, Vec3] = {
    "L_Wrist": _PZ,
    "L_ThumbMetacarpal": _NZ,
    "L_ThumbProximal": _PZ,
    "L_ThumbDistal": _PZ,
    "L_IndexMetacarpal": _PZ,
    "L_IndexProximal": _NZ,
    "L_IndexIntermediate": _NZ,
    "L_IndexDistal": _NZ,
    "L_MiddleMetacarpal": _PZ,
    "L_MiddleProximal": _NZ,
    "L_MiddleIntermediate": _NZ,
    "L_MiddleDistal": _NZ,
    "L_RingMetacarpal": _PZ,
    "L_RingProximal": _NZ,
    "L_RingIntermediate": _NZ,
    "L_RingDistal": _NZ,
    "L_LittleMetacarpal": _NX,
    "L_LittleProximal": _NZ,
    "L_LittleIntermediate": _NZ,
    "L_LittleDistal": _NZ,
}

RIGHT_FORWARD_AXIS: Dict[str, Vec3] = {
    "R_Wrist": _PZ,
    "R_ThumbMetacarpal": _NZ,
    "R_ThumbProximal": _PY,
    "R_ThumbDistal": _PY,
    "R_IndexMetacarpal": _PZ,
    "R_IndexProximal": _NZ,
    "R_IndexIntermediate": _NZ,
    "R_IndexDistal": _NZ,
    "R_MiddleMetacarpal": _PZ,
    "R_MiddleProximal": _NZ,
    "R_MiddleIntermediate": _NZ,
    "R_MiddleDistal": _NZ,
    "R_RingMetacarpal": _PZ,
    "R_RingProximal": _NZ,
    "R_RingIntermediate": _NZ,
    "R_RingDistal": _NZ,
    "R_LittleMetacarpal": _PX,
    "R_LittleProximal": _NZ,
    "R_LittleIntermediate": _NZ,
    "R_LittleDistal": _NZ,
}


def _mirror_bone_map(left_map: Dict[str, float | str]) -> Dict[str, float | str]:
    return {name.replace("L_", "R_", 1): value for name, value in left_map.items()}


# Static per-bone pose variants derived from repeatable auto-calibration results for this FBX rig.
LEFT_AXIS_SIGN: Dict[str, float] = {
    "L_Wrist": 1.0,
    "L_ThumbMetacarpal": -1.0,
    "L_ThumbProximal": 1.0,
    "L_ThumbDistal": 1.0,
    "L_IndexMetacarpal": 1.0,
    "L_IndexProximal": -1.0,
    "L_IndexIntermediate": -1.0,
    "L_IndexDistal": 1.0,
    "L_MiddleMetacarpal": 1.0,
    "L_MiddleProximal": -1.0,
    "L_MiddleIntermediate": 1.0,
    "L_MiddleDistal": 1.0,
    "L_RingMetacarpal": 1.0,
    "L_RingProximal": -1.0,
    "L_RingIntermediate": 1.0,
    "L_RingDistal": 1.0,
    "L_LittleMetacarpal": 1.0,
    "L_LittleProximal": -1.0,
    "L_LittleIntermediate": 1.0,
    "L_LittleDistal": 1.0,
}
RIGHT_AXIS_SIGN: Dict[str, float] = _mirror_bone_map(LEFT_AXIS_SIGN)

LEFT_AXIS_ORDER: Dict[str, str] = {name: "r*rest" for name in LEFT_AXIS_SIGN.keys()}
RIGHT_AXIS_ORDER: Dict[str, str] = _mirror_bone_map(LEFT_AXIS_ORDER)


class FBXHandRig:
    def __init__(
        self,
        actor: Actor,
        actor_root: NodePath,
        is_left: bool,
        pos_smooth: float,
        rot_smooth: float,
        wrist_stability: float,
        model_scale: float,
        landmark_scale: float,
        root_motion_scale: float,
        bone_forward: Vec3,
        forward_mode: str = "table",
        axis_strategy: str = "static",
        finger_follow_hand: bool = False,
        axis_calibration_frames: int = 6,
        allow_runtime_axis_fallback: bool = False,
        debug_dump: bool = False,
        debug_max_frames: int = 2,
        debug_bones: Set[str] | None = None,
    ) -> None:
        self.actor = actor
        self.actor_root = actor_root
        self.is_left = is_left
        self.pos_smooth = max(0.0, min(0.98, pos_smooth))
        self.rot_smooth = max(0.0, min(0.98, rot_smooth))
        self.wrist_stability = max(0.0, min(0.98, wrist_stability))
        self.model_scale = max(0.001, model_scale)
        self.landmark_scale = max(0.1, landmark_scale)
        self.root_motion_scale = max(0.0, root_motion_scale)
        self.debug_dump = bool(debug_dump)
        self.debug_max_frames = max(0, int(debug_max_frames))
        self.debug_bones = debug_bones or set()
        self.forward_mode = forward_mode
        self.axis_strategy = axis_strategy
        self.finger_follow_hand = finger_follow_hand
        self.axis_calibration_frames = max(1, int(axis_calibration_frames))
        self.allow_runtime_axis_fallback = allow_runtime_axis_fallback

        self.mapping = LANDMARK_TO_BONE_SEQ_LEFT if is_left else LANDMARK_TO_BONE_SEQ_RIGHT
        self.bone_forward = Vec3(bone_forward)
        self.joints: Dict[str, NodePath] = {}
        self.rest_quat: Dict[str, Quat] = {}
        self.prev_hpr: Dict[str, Vec3] = {}
        self.local_fwd: Dict[str, Vec3] = {}
        self.axis_sign: Dict[str, float] = {}
        self.axis_order: Dict[str, str] = {}
        self.axis_calibrated = axis_strategy != "auto"
        self.axis_calibration_seen = 0
        self.axis_scores: Dict[str, Dict[Tuple[float, str], List[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [0.0, 0.0])
        )

        self.smoothed_pts: List[Vec3] = []
        self.root_pos = Vec3(0, 0, 0)
        self.palm_up_smooth = Vec3(0, 1, 0)
        self.palm_normal_smooth = Vec3(0, 0, 1)

        all_bones = set()
        all_bones.add(self.mapping["wrist"][1])
        for k in ("thumb", "index", "middle", "ring", "little"):
            for _, name, _ in self.mapping[k]:
                all_bones.add(name)

        for bone in sorted(all_bones):
            node = actor.controlJoint(None, "modelRoot", bone)
            if node is not None and not node.isEmpty():
                self.joints[bone] = node

        self.actor.reparentTo(self.actor_root)
        self.actor.setScale(1.0)
        self.actor_root.setScale(self.model_scale)

        for bone, node in self.joints.items():
            q = Quat(node.getQuat(self.actor_root))
            self.rest_quat[bone] = q
            hpr = node.getHpr(self.actor_root)
            self.prev_hpr[bone] = Vec3(hpr.x, hpr.y, hpr.z)

        self._init_local_forward_axes()
        self._init_rest_palm_frame()
        self._init_axis_variants()
        if self.debug_dump:
            side = "Left" if self.is_left else "Right"
            print(
                f"[debug][{side}] init bone_forward={tuple(round(v,4) for v in (self.bone_forward.x, self.bone_forward.y, self.bone_forward.z))} "
                f"axis_strategy={self.axis_strategy} runtime_fallback={self.allow_runtime_axis_fallback}",
                flush=True,
            )
            for bone in sorted(self.local_fwd.keys()):
                lv = self.local_fwd[bone]
                print(f"[debug][{side}] local_fwd {bone} = ({lv.x:.4f}, {lv.y:.4f}, {lv.z:.4f})", flush=True)

    def _init_axis_variants(self) -> None:
        sign_table = LEFT_AXIS_SIGN if self.is_left else RIGHT_AXIS_SIGN
        order_table = LEFT_AXIS_ORDER if self.is_left else RIGHT_AXIS_ORDER
        for bone in self.joints.keys():
            self.axis_sign[bone] = float(sign_table.get(bone, 1.0))
            self.axis_order[bone] = str(order_table.get(bone, "r*rest"))

    def _init_local_forward_axes(self) -> None:
        if self.forward_mode == "table":
            table = LEFT_FORWARD_AXIS if self.is_left else RIGHT_FORWARD_AXIS
            fallback = Vec3(self.bone_forward)
            if fallback.lengthSquared() < 1e-8:
                fallback = Vec3(0, 1, 0)
            else:
                fallback.normalize()
            for bone in self.joints.keys():
                v = table.get(bone, fallback)
                vv = Vec3(v)
                if vv.lengthSquared() < 1e-8:
                    vv = Vec3(0, 1, 0)
                else:
                    vv.normalize()
                self.local_fwd[bone] = vv
            return

        # Uniform mode.
        fwd = Vec3(self.bone_forward)
        if fwd.lengthSquared() < 1e-8:
            fwd = Vec3(0, 1, 0)
        else:
            fwd.normalize()
        for bone in self.joints.keys():
            self.local_fwd[bone] = Vec3(fwd)

    def _init_rest_palm_frame(self) -> None:
        wrist_name = self.mapping["wrist"][1]
        mid_meta = self.mapping["middle"][0][1]
        idx_meta = self.mapping["index"][0][1]
        lit_meta = self.mapping["little"][0][1]

        w_pos = self.joints[wrist_name].getPos(self.actor_root)
        m_pos = self.joints[mid_meta].getPos(self.actor_root)
        i_pos = self.joints[idx_meta].getPos(self.actor_root)
        l_pos = self.joints[lit_meta].getPos(self.actor_root)

        rest_up = m_pos - w_pos
        if rest_up.lengthSquared() < 1e-8:
            rest_up = Vec3(0, 1, 0)
        else:
            rest_up.normalize()

        rest_side = l_pos - i_pos
        if rest_side.lengthSquared() < 1e-8:
            rest_side = Vec3(1, 0, 0)
        else:
            rest_side.normalize()

        rest_norm = rest_side.cross(rest_up)
        if rest_norm.lengthSquared() < 1e-8:
            rest_norm = Vec3(0, 0, 1)
        else:
            rest_norm.normalize()
        if self.is_left:
            rest_norm = -rest_norm

        self.rest_dummy = self.actor_root.attachNewNode("rest_dummy")
        self.curr_dummy = self.actor_root.attachNewNode("curr_dummy")
        self.rest_dummy.setPos(0, 0, 0)
        self.rest_dummy.lookAt(self.rest_dummy.getPos() + rest_up, rest_norm)
        self.rest_palm_q = Quat(self.rest_dummy.getQuat())

    def set_visible(self, visible: bool) -> None:
        if visible:
            self.actor_root.show()
        else:
            self.actor_root.hide()

    def drive_from_landmarks(self, landmarks: List[Dict], x_offset: float, frame_idx: int = -1) -> bool:
        if len(landmarks) < 21:
            return False
        self.set_visible(True)

        pts_raw = [self._lm_to_local(lm, x_offset) for lm in landmarks]
        if not self.smoothed_pts:
            self.smoothed_pts = list(pts_raw)
        else:
            a = 1.0 - self.pos_smooth
            for i, p in enumerate(pts_raw):
                self.smoothed_pts[i] = self._lerp_vec3(self.smoothed_pts[i], p, a)
        pts = self.smoothed_pts

        target_root = pts[0] * self.root_motion_scale
        self.root_pos = self._lerp_vec3(self.root_pos, target_root, 1.0 - self.pos_smooth)
        self.actor_root.setPos(self.root_pos)
        self.actor_root.setScale(self.model_scale)

        curr_up = pts[9] - pts[0]
        if curr_up.lengthSquared() < 1e-8:
            curr_up = Vec3(0, 1, 0)
        else:
            curr_up.normalize()

        curr_side = pts[17] - pts[5]
        if curr_side.lengthSquared() < 1e-8:
            curr_side = Vec3(1, 0, 0)
        else:
            curr_side.normalize()

        curr_norm = curr_side.cross(curr_up)
        if curr_norm.lengthSquared() < 1e-8:
            curr_norm = Vec3(0, 0, 1)
        else:
            curr_norm.normalize()
        if self.is_left:
            curr_norm = -curr_norm

        ws = 1.0 - self.wrist_stability
        self.palm_up_smooth = self._lerp_vec3(self.palm_up_smooth, curr_up, ws)
        self.palm_normal_smooth = self._lerp_vec3(self.palm_normal_smooth, curr_norm, ws)
        if self.palm_up_smooth.lengthSquared() > 1e-8:
            self.palm_up_smooth.normalize()
        else:
            self.palm_up_smooth = Vec3(0, 1, 0)
        if self.palm_normal_smooth.lengthSquared() > 1e-8:
            self.palm_normal_smooth.normalize()
        else:
            self.palm_normal_smooth = Vec3(0, 0, 1)

        self.curr_dummy.setPos(0, 0, 0)
        self.curr_dummy.lookAt(self.curr_dummy.getPos() + self.palm_up_smooth, self.palm_normal_smooth)
        curr_palm_q = Quat(self.curr_dummy.getQuat())
        r_hand = curr_palm_q * self.rest_palm_q.conjugate()

        if not self.axis_calibrated:
            self._auto_calibrate_axes(pts, r_hand, frame_idx)

        do_debug_frame = self.debug_dump and (frame_idx < 0 or frame_idx < self.debug_max_frames)
        if do_debug_frame:
            side = "Left" if self.is_left else "Right"
            rp = self.root_pos
            print(
                f"[debug][{side}][frame {frame_idx}] root=({rp.x:.4f},{rp.y:.4f},{rp.z:.4f}) "
                f"palm_up=({self.palm_up_smooth.x:.4f},{self.palm_up_smooth.y:.4f},{self.palm_up_smooth.z:.4f}) "
                f"palm_n=({self.palm_normal_smooth.x:.4f},{self.palm_normal_smooth.y:.4f},{self.palm_normal_smooth.z:.4f}) "
                f"r_hand_hpr={r_hand.getHpr()}"
            , flush=True)

        wrist_name = self.mapping["wrist"][1]
        self._drive_bone(wrist_name, r_hand, pts[0], pts[9], frame_idx)
        for chain_name in ("thumb", "index", "middle", "ring", "little"):
            for lm_idx, bone_name, child_idx in self.mapping[chain_name]:
                self._drive_bone(bone_name, r_hand, pts[lm_idx], pts[child_idx], frame_idx)
        return True

    def _drive_bone(self, bone_name: str, r_hand: Quat, p: Vec3, c: Vec3, frame_idx: int = -1) -> None:
        joint = self.joints.get(bone_name)
        if joint is None:
            return
        target_dir = c - p
        if target_dir.lengthSquared() < 1e-8:
            return
        target_dir.normalize()

        local_fwd = self.local_fwd.get(bone_name, Vec3(0, 0, 1))
        sign = self.axis_sign.get(bone_name, 1.0)
        local_fwd = local_fwd * sign
        if "Wrist" in bone_name or self.finger_follow_hand:
            hand_q = r_hand
        else:
            hand_q = Quat.identQuat()

        preferred_sign = self.axis_sign.get(bone_name, 1.0)
        preferred_order = self.axis_order.get(bone_name, "r*rest")
        base_q, curr_fwd, base_order, base_dot, sign = self._choose_pose_variant(
            bone_name=bone_name,
            hand_q=hand_q,
            target_dir=target_dir,
            preferred_sign=preferred_sign,
            preferred_order=preferred_order,
            strict_preferred=True,
        )
        if base_q is None:
            return
        # Runtime safeguard: if preferred axis becomes opposite to target direction, fallback to best current variant.
        if self.allow_runtime_axis_fallback and base_dot < 0.10:
            best_q, best_fwd, best_order, best_dot, best_sign = self._choose_pose_variant(
                bone_name=bone_name,
                hand_q=hand_q,
                target_dir=target_dir,
                preferred_sign=preferred_sign,
                preferred_order=preferred_order,
                strict_preferred=False,
            )
            if best_q is not None and best_dot > base_dot + 0.20:
                prev_dot = base_dot
                base_q, curr_fwd, base_order, base_dot, sign = best_q, best_fwd, best_order, best_dot, best_sign
                if self.debug_dump and (frame_idx < 0 or frame_idx < self.debug_max_frames):
                    side = "Left" if self.is_left else "Right"
                    print(
                        f"[debug][{side}][frame {frame_idx}] fallback bone={bone_name} "
                        f"from(sign={preferred_sign:+.0f},order={preferred_order},dot={prev_dot:.3f}) "
                        f"to(sign={best_sign:+.0f},order={best_order},dot={best_dot:.3f})",
                        flush=True,
                    )
        # Low-confidence direction alignment guard: avoid full-angle snaps when base axis is weak.
        swing_mix = max(0.0, min(1.0, (base_dot - 0.20) / 0.60))
        if "Metacarpal" in bone_name:
            swing_mix = min(swing_mix, 0.45)
        if bone_name.endswith("ThumbMetacarpal") or bone_name.endswith("LittleMetacarpal"):
            swing_mix = min(swing_mix, 0.30)
        blend_dir = self._lerp_vec3(curr_fwd, target_dir, swing_mix)
        if blend_dir.lengthSquared() < 1e-8:
            blend_dir = Vec3(curr_fwd)
        else:
            blend_dir.normalize()
        swing_q = self._shortest_arc(curr_fwd, blend_dir)
        target_q = swing_q * base_q

        alpha = 1.0 - self.rot_smooth
        if "Wrist" in bone_name:
            alpha *= (1.0 - 0.7 * self.wrist_stability)
        alpha = max(0.02, min(1.0, alpha))

        target_hpr = target_q.getHpr()
        prev_hpr = self.prev_hpr.get(bone_name, target_hpr)
        smoothed_hpr = Vec3(
            self._lerp_angle(prev_hpr.x, target_hpr.x, alpha),
            self._lerp_angle(prev_hpr.y, target_hpr.y, alpha),
            self._lerp_angle(prev_hpr.z, target_hpr.z, alpha),
        )
        self.prev_hpr[bone_name] = smoothed_hpr
        smoothed_q = Quat()
        smoothed_q.setHpr(smoothed_hpr)
        joint.setQuat(self.actor_root, smoothed_q)

        do_debug_frame = self.debug_dump and (frame_idx < 0 or frame_idx < self.debug_max_frames)
        do_debug_bone = (not self.debug_bones) or (bone_name in self.debug_bones)
        if do_debug_frame and do_debug_bone:
            side = "Left" if self.is_left else "Right"
            print(
                f"[debug][{side}][frame {frame_idx}] bone={bone_name} "
                f"alpha={alpha:.3f} "
                f"local_fwd=({self.local_fwd[bone_name].x:.3f},{self.local_fwd[bone_name].y:.3f},{self.local_fwd[bone_name].z:.3f}) "
                f"sign={sign:+.0f} "
                f"target_dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f}) "
                f"curr_fwd=({curr_fwd.x:.3f},{curr_fwd.y:.3f},{curr_fwd.z:.3f}) "
                f"swing_mix={swing_mix:.3f} "
                f"base_order={base_order} base_dot={base_dot:.3f} "
                f"base_hpr={base_q.getHpr()} swing_hpr={swing_q.getHpr()} "
                f"target_hpr={target_hpr} smoothed_hpr={smoothed_hpr}"
            , flush=True)

    def _auto_calibrate_axes(self, pts: List[Vec3], r_hand: Quat, frame_idx: int) -> None:
        side = "Left" if self.is_left else "Right"
        if self.debug_dump and self.axis_calibration_seen == 0:
            print(
                f"[debug][{side}][frame {frame_idx}] auto-calibrate axis/order start "
                f"(target_frames={self.axis_calibration_frames})",
                flush=True,
            )
        used_any = False
        for chain_name in ("wrist", "thumb", "index", "middle", "ring", "little"):
            if chain_name == "wrist":
                items = [self.mapping["wrist"]]
            else:
                items = self.mapping[chain_name]
            for lm_idx, bone_name, child_idx in items:
                if bone_name not in self.joints:
                    continue
                tdir = pts[child_idx] - pts[lm_idx]
                if tdir.lengthSquared() < 1e-8:
                    continue
                tdir.normalize()
                used_any = True
                if "Wrist" in bone_name or self.finger_follow_hand:
                    hand_q = r_hand
                else:
                    hand_q = Quat.identQuat()
                rest_q = self.rest_quat[bone_name]
                for sign in (1.0, -1.0):
                    lf = self.local_fwd.get(bone_name, Vec3(0, 0, 1)) * sign
                    qa = hand_q * rest_q
                    qb = rest_q * hand_q
                    fa = qa.xform(lf)
                    fb = qb.xform(lf)
                    if fa.lengthSquared() > 1e-8:
                        fa.normalize()
                        da = fa.dot(tdir)
                        rec = self.axis_scores[bone_name][(sign, "r*rest")]
                        rec[0] += da
                        rec[1] += 1.0
                    if fb.lengthSquared() > 1e-8:
                        fb.normalize()
                        db = fb.dot(tdir)
                        rec = self.axis_scores[bone_name][(sign, "rest*r")]
                        rec[0] += db
                        rec[1] += 1.0
        if not used_any:
            return
        self.axis_calibration_seen += 1
        if self.axis_calibration_seen < self.axis_calibration_frames:
            if self.debug_dump and (frame_idx < 0 or frame_idx < self.debug_max_frames):
                print(
                    f"[debug][{side}][frame {frame_idx}] auto-cal progress "
                    f"{self.axis_calibration_seen}/{self.axis_calibration_frames}",
                    flush=True,
                )
            return
        for bone_name, options in self.axis_scores.items():
            best_score = None
            best_sign = self.axis_sign.get(bone_name, 1.0)
            best_order = self.axis_order.get(bone_name, "r*rest")
            for (sign, order), (sum_dot, cnt) in options.items():
                if cnt <= 0.0:
                    continue
                avg = sum_dot / cnt
                if best_score is None or avg > best_score:
                    best_score = avg
                    best_sign = sign
                    best_order = order
            self.axis_sign[bone_name] = best_sign
            self.axis_order[bone_name] = best_order
            if self.debug_dump:
                print(
                    f"[debug][{side}] auto-cal locked bone={bone_name} "
                    f"avg_dot={best_score if best_score is not None else -9.0:.3f} "
                    f"sign={best_sign:+.0f} order={best_order}",
                    flush=True,
                )
        self.axis_calibrated = True
        if self.debug_dump:
            print(f"[debug][{side}] auto-calibrate axis/order done", flush=True)

    def _choose_pose_variant(
        self,
        bone_name: str,
        hand_q: Quat,
        target_dir: Vec3,
        preferred_sign: float,
        preferred_order: str,
        strict_preferred: bool,
    ):
        rest_q = self.rest_quat[bone_name]
        local_fwd_base = self.local_fwd.get(bone_name, Vec3(0, 0, 1))
        candidates = []
        for sign in (1.0, -1.0):
            lf = local_fwd_base * sign
            qa = hand_q * rest_q
            qb = rest_q * hand_q
            fa = qa.xform(lf)
            fb = qb.xform(lf)
            if fa.lengthSquared() > 1e-8:
                fa.normalize()
                candidates.append((qa, fa, "r*rest", fa.dot(target_dir), sign))
            if fb.lengthSquared() > 1e-8:
                fb.normalize()
                candidates.append((qb, fb, "rest*r", fb.dot(target_dir), sign))
        if not candidates:
            return None, None, "", -1.0, preferred_sign
        if strict_preferred:
            for item in candidates:
                _, _, order, _, sign = item
                if sign == preferred_sign and order == preferred_order:
                    return item
        return max(candidates, key=lambda it: it[3])

    def _lm_to_local(self, lm: Dict, x_offset: float) -> Vec3:
        x = (float(lm["x"]) - 0.5) * self.landmark_scale
        z = (-(float(lm["y"]) - 0.5)) * self.landmark_scale
        y = float(lm["z"]) * self.landmark_scale
        if self.is_left:
            x = x + x_offset
        else:
            x = -x - x_offset
        return Vec3(x, y, z)

    @staticmethod
    def _shortest_arc(v1: Vec3, v2: Vec3) -> Quat:
        a = Vec3(v1)
        b = Vec3(v2)
        if a.lengthSquared() < 1e-8 or b.lengthSquared() < 1e-8:
            return Quat.identQuat()
        a.normalize()
        b.normalize()
        cross = a.cross(b)
        dot = max(-1.0, min(1.0, a.dot(b)))
        if cross.lengthSquared() < 1e-8:
            if dot > 0.0:
                return Quat.identQuat()
            perp = Vec3(1, 0, 0) if abs(a.x) < 0.9 else Vec3(0, 1, 0)
            axis = a.cross(perp)
            if axis.lengthSquared() < 1e-8:
                axis = Vec3(0, 0, 1)
            else:
                axis.normalize()
            q180 = Quat()
            q180.setFromAxisAngle(180.0, axis)
            return q180
        cross.normalize()
        q = Quat()
        q.setFromAxisAngle(math.degrees(math.acos(dot)), cross)
        return q

    @staticmethod
    def _delta_angle(a: float, b: float) -> float:
        d = a - b
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        return d

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        return a + FBXHandRig._delta_angle(b, a) * t

    @staticmethod
    def _lerp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
        return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)


class FBXSkinningViewer(ShowBase):
    def __init__(
        self,
        stream_path: Path,
        left_fbx: Path,
        right_fbx: Path,
        fps: float,
        hand_separation: float,
        hide_after_miss: int,
        model_scale: float,
        landmark_scale: float,
        root_motion_scale: float,
        pos_smooth: float,
        rot_smooth: float,
        wrist_stability: float,
        bone_forward: Vec3,
        forward_mode: str,
        axis_strategy: str,
        finger_follow_hand: bool,
        allow_runtime_axis_fallback: bool,
        debug_dump: bool,
        debug_max_frames: int,
        debug_bones: Set[str],
    ) -> None:
        super().__init__()
        self.disableMouse()
        self.frames = [json.loads(l) for l in stream_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.frames:
            raise RuntimeError(f"No frame data in {stream_path}")

        self.play_interval = 1.0 / max(1.0, fps)
        self.time_acc = 0.0
        self.current_frame = 0
        self.playing = True
        self.hand_separation = hand_separation
        self.hide_after_miss = max(1, hide_after_miss)
        self.freeze_on_miss = True
        self.left_miss = 0
        self.right_miss = 0
        self.cam_y_min = -6.0
        self.cam_y_max = -0.20
        self.debug_dump = debug_dump
        self.debug_max_frames = debug_max_frames

        self._init_window()
        self._init_scene()

        self.left_actor = Actor(_to_model_path(left_fbx))
        self.right_actor = Actor(_to_model_path(right_fbx))
        self.left_rig = FBXHandRig(
            self.left_actor,
            self.render.attachNewNode("left_actor_root"),
            is_left=True,
            pos_smooth=pos_smooth,
            rot_smooth=rot_smooth,
            wrist_stability=wrist_stability,
            model_scale=model_scale,
            landmark_scale=landmark_scale,
            root_motion_scale=root_motion_scale,
            bone_forward=bone_forward,
            forward_mode=forward_mode,
            axis_strategy=axis_strategy,
            finger_follow_hand=finger_follow_hand,
            allow_runtime_axis_fallback=allow_runtime_axis_fallback,
            debug_dump=debug_dump,
            debug_max_frames=debug_max_frames,
            debug_bones=debug_bones,
        )
        self.right_rig = FBXHandRig(
            self.right_actor,
            self.render.attachNewNode("right_actor_root"),
            is_left=False,
            pos_smooth=pos_smooth,
            rot_smooth=rot_smooth,
            wrist_stability=wrist_stability,
            model_scale=model_scale,
            landmark_scale=landmark_scale,
            root_motion_scale=root_motion_scale,
            bone_forward=bone_forward,
            forward_mode=forward_mode,
            axis_strategy=axis_strategy,
            finger_follow_hand=finger_follow_hand,
            allow_runtime_axis_fallback=allow_runtime_axis_fallback,
            debug_dump=debug_dump,
            debug_max_frames=debug_max_frames,
            debug_bones=debug_bones,
        )
        self.left_rig.set_visible(False)
        self.right_rig.set_visible(False)

        self._init_controls()
        self.taskMgr.add(self._update_task, "update_task")
        self._apply_frame(0)

    def _init_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Panda3D FBX Skinning Hand Viewer")
        props.setSize(1520, 960)
        self.win.requestProperties(props)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.03, 0.03, 0.04, 1.0)
        self.camLens.setNearFar(0.02, 200.0)

    def _init_scene(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.33, 0.33, 0.35, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        key = DirectionalLight("key")
        key.setColor(Vec4(0.96, 0.95, 0.93, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-30, -35, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.42, 0.47, 0.56, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(130, -18, 0)
        self.render.setLight(fill_np)

        mat = Material()
        mat.setShininess(70.0)
        mat.setSpecular((0.34, 0.34, 0.34, 1.0))
        self.render.setMaterial(mat, 1)

        self.cam_pivot = self.render.attachNewNode("cam_pivot")
        self.camera.reparentTo(self.cam_pivot)
        self.camera.setPos(0, -3.5, 1.1)
        self.camera.lookAt(0, 0, 0)

    def _init_controls(self) -> None:
        self.accept("space", self._toggle_play)
        self.accept("arrow_right", self._step_next)
        self.accept("arrow_left", self._step_prev)
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("=", self._zoom_in)
        self.accept("+", self._zoom_in)
        self.accept("-", self._zoom_out)
        self.accept("_", self._zoom_out)
        self.accept("r", self._reset_camera)
        self.accept("escape", self.userExit)

        self.dragging = False
        self.last_mouse = Point3(0, 0, 0)
        self.accept("mouse1", self._drag_start)
        self.accept("mouse1-up", self._drag_end)
        self.taskMgr.add(self._drag_task, "drag_task")

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    def _step_next(self) -> None:
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _step_prev(self) -> None:
        self.current_frame = (self.current_frame - 1 + len(self.frames)) % len(self.frames)
        self._apply_frame(self.current_frame)

    def _zoom_in(self) -> None:
        y = self.camera.getY() + 0.28
        self.camera.setY(min(self.cam_y_max, y))

    def _zoom_out(self) -> None:
        y = self.camera.getY() - 0.28
        self.camera.setY(max(self.cam_y_min, y))

    def _reset_camera(self) -> None:
        self.cam_pivot.setHpr(0, 0, 0)
        self.camera.setPos(0, -3.5, 1.1)
        self.camera.lookAt(0, 0, 0)

    def _drag_start(self) -> None:
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = Point3(m.getX(), m.getY(), 0)
            self.dragging = True

    def _drag_end(self) -> None:
        self.dragging = False

    def _drag_task(self, task: Task):
        if not self.dragging or not self.mouseWatcherNode.hasMouse():
            return Task.cont
        m = self.mouseWatcherNode.getMouse()
        dx = m.getX() - self.last_mouse.x
        dy = m.getY() - self.last_mouse.y
        self.last_mouse = Point3(m.getX(), m.getY(), 0)
        self.cam_pivot.setH(self.cam_pivot.getH() - dx * 135.0)
        self.cam_pivot.setP(max(-88.0, min(88.0, self.cam_pivot.getP() + dy * 90.0)))
        return Task.cont

    def _update_task(self, task: Task):
        if not self.playing:
            return Task.cont
        self.time_acc += globalClock.getDt()
        while self.time_acc >= self.play_interval:
            self.time_acc -= self.play_interval
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self._apply_frame(self.current_frame)
        return Task.cont

    def _apply_frame(self, frame_idx: int) -> None:
        frame = self.frames[frame_idx]
        hands = frame.get("hands", [])
        if self.debug_dump and frame_idx < self.debug_max_frames:
            htypes = [h.get("hand_type", "?") for h in hands]
            print(f"[debug][frame {frame_idx}] hands={len(hands)} types={htypes}", flush=True)
        left = None
        right = None
        for h in hands:
            ht = h.get("hand_type")
            if ht == "Left":
                left = h
            elif ht == "Right":
                right = h

        if left is None:
            self.left_miss += 1
            if not self.freeze_on_miss and self.left_miss >= self.hide_after_miss:
                self.left_rig.set_visible(False)
        else:
            ok = self.left_rig.drive_from_landmarks(
                left.get("landmarks", []),
                x_offset=-self.hand_separation,
                frame_idx=frame_idx,
            )
            if ok:
                self.left_miss = 0
            else:
                self.left_miss += 1
                if not self.freeze_on_miss and self.left_miss >= self.hide_after_miss:
                    self.left_rig.set_visible(False)

        if right is None:
            self.right_miss += 1
            if not self.freeze_on_miss and self.right_miss >= self.hide_after_miss:
                self.right_rig.set_visible(False)
        else:
            ok = self.right_rig.drive_from_landmarks(
                right.get("landmarks", []),
                x_offset=self.hand_separation,
                frame_idx=frame_idx,
            )
            if ok:
                self.right_miss = 0
            else:
                self.right_miss += 1
                if not self.freeze_on_miss and self.right_miss >= self.hide_after_miss:
                    self.right_rig.set_visible(False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FBX skinning-driven Panda3D hand viewer")
    p.add_argument("--stream", required=True, help="unity_gesture_stream.jsonl path")
    p.add_argument("--left-fbx", default="../new_sign_language/LeftHand.fbx")
    p.add_argument("--right-fbx", default="../new_sign_language/RightHand.fbx")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--hand-separation", type=float, default=0.22)
    p.add_argument("--hide-after-miss", type=int, default=8)
    p.add_argument("--model-scale", type=float, default=0.018)
    p.add_argument("--landmark-scale", type=float, default=13.5)
    p.add_argument("--root-motion-scale", type=float, default=0.18)
    p.add_argument("--pos-smooth", type=float, default=0.72)
    p.add_argument("--rot-smooth", type=float, default=0.82)
    p.add_argument("--wrist-stability", type=float, default=0.65)
    p.add_argument(
        "--bone-forward",
        default="0,1,0",
        help="Uniform local forward axis for all bones: x,y,z (e.g. 0,1,0 or 0,-1,0 or 0,0,1)",
    )
    p.add_argument(
        "--forward-mode",
        choices=["table", "uniform"],
        default="table",
        help="Use measured per-bone axis table or one uniform axis for all bones",
    )
    p.add_argument(
        "--axis-strategy",
        choices=["static", "auto"],
        default="static",
        help="Static uses rig-specific locked sign/order defaults; auto re-learns them at runtime",
    )
    p.add_argument(
        "--finger-follow-hand",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, apply palm global rotation to finger base orientation too",
    )
    p.add_argument(
        "--allow-runtime-axis-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow per-frame sign/order fallback when the preferred axis variant points away from the target",
    )
    p.add_argument("--debug-dump", action="store_true", help="Print detailed per-frame and per-bone debug data")
    p.add_argument("--debug-max-frames", type=int, default=2, help="How many initial frames to print in debug mode")
    p.add_argument(
        "--debug-bones",
        default="",
        help="Comma-separated bone names to print; empty means all bones",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        bx, by, bz = [float(v.strip()) for v in args.bone_forward.split(",")]
        bone_forward = Vec3(bx, by, bz)
    except Exception as e:
        raise SystemExit(f"invalid --bone-forward '{args.bone_forward}': {e}")
    debug_bones: Set[str] = set()
    if args.debug_bones.strip():
        debug_bones = {b.strip() for b in args.debug_bones.split(",") if b.strip()}
    if args.debug_dump:
        print(
            "[debug][startup] "
            f"stream={Path(args.stream).resolve()} "
            f"left_fbx={Path(args.left_fbx).resolve()} "
            f"right_fbx={Path(args.right_fbx).resolve()} "
            f"fps={args.fps} hand_separation={args.hand_separation} hide_after_miss={args.hide_after_miss} "
            f"model_scale={args.model_scale} landmark_scale={args.landmark_scale} "
            f"root_motion_scale={args.root_motion_scale} pos_smooth={args.pos_smooth} "
            f"rot_smooth={args.rot_smooth} wrist_stability={args.wrist_stability} "
            f"forward_mode={args.forward_mode} axis_strategy={args.axis_strategy} "
            f"allow_runtime_axis_fallback={args.allow_runtime_axis_fallback} "
            f"finger_follow_hand={args.finger_follow_hand} "
            f"bone_forward={args.bone_forward} debug_max_frames={args.debug_max_frames} "
            f"debug_bones={sorted(debug_bones) if debug_bones else 'ALL'}"
        , flush=True)
    app = FBXSkinningViewer(
        stream_path=Path(args.stream).resolve(),
        left_fbx=Path(args.left_fbx).resolve(),
        right_fbx=Path(args.right_fbx).resolve(),
        fps=args.fps,
        hand_separation=args.hand_separation,
        hide_after_miss=args.hide_after_miss,
        model_scale=args.model_scale,
        landmark_scale=args.landmark_scale,
        root_motion_scale=args.root_motion_scale,
        pos_smooth=args.pos_smooth,
        rot_smooth=args.rot_smooth,
        wrist_stability=args.wrist_stability,
        bone_forward=bone_forward,
        forward_mode=args.forward_mode,
        axis_strategy=args.axis_strategy,
        finger_follow_hand=args.finger_follow_hand,
        allow_runtime_axis_fallback=args.allow_runtime_axis_fallback,
        debug_dump=args.debug_dump,
        debug_max_frames=args.debug_max_frames,
        debug_bones=debug_bones,
    )
    app.run()


if __name__ == "__main__":
    main()
