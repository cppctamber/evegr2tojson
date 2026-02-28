bl_info = {
    "name": "GR2 Importer (via EXE + JSON)",
    "author": "CPPC T'amber",
    "version": (0, 0, 7),
    "blender": (5, 0, 0),
    "location": "File > Import",
    "description": "Imports GR2 and GR2.JSON files using an external exe",
    "category": "Import-Export",
}

import os
import json
import math
import struct
import subprocess
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import bpy
from bpy.types import Operator, AddonPreferences
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
)
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector, Quaternion


# =============================================================================
# Preferences
# =============================================================================

class GR2ImporterPreferences(AddonPreferences):
    bl_idname = __name__

    exe_path: StringProperty(
        name="evegr2tojson file path",
        description="Path to your evegr2tojson EXE. Must output <input_without_ext>.gr2_json next to the input.",
        subtype="FILE_PATH",
        default="",
    )

    show_console_windows: BoolProperty(
        name="Show console window (Windows)",
        description="If disabled on Windows, tries to hide the EXE console window when running the evegr2tojson.",
        default=False,
    )

    import_scale: FloatProperty(
        name="Import scale",
        description="Global scale applied to imported objects.",
        default=0.01,
        min=1e-6,
        soft_max=1000.0,
    )

    import_rot_x_deg: FloatProperty(
        name="Import rotation X (degrees)",
        description="Global rotation around X applied to imported objects.",
        default=90.0,
        soft_min=-360.0,
        soft_max=360.0,
    )

    # UVs
    flip_uv_v_default: BoolProperty(
        name="Flip UV V (Y) (default)",
        description="Flips V (Y) coordinate for UV layers on import.",
        default=True,
    )

    # Smoothing groups (Auto Smooth-style)
    use_smoothing_groups_default: BoolProperty(
        name="Use smoothing groups (default)",
        description="Applies Smooth shading + Auto Smooth (angle) to imported meshes.",
        default=True,
    )

    smoothing_angle_default: FloatProperty(
        name="Smoothing angle (degrees)",
        description="Angle threshold for Auto Smooth (0–180). Higher = smoother.",
        default=180.0,
        min=0.0,
        max=180.0,
    )

    apply_skinning_default: BoolProperty(
        name="Apply Skinning (default)",
        description="Default for Apply Skinning on import operators.",
        default=True,
    )

    import_anims_default: BoolProperty(
        name="Import Animations (default)",
        description="Default for Import Animations on import operators.",
        default=True,
    )

    resample_anims: BoolProperty(
        name="Resample animations (recommended)",
        description="Samples curves at fixed dt (timeStep/oversampling) and keys every sample to avoid quaternion component interpolation artifacts.",
        default=False,
    )

    max_keys_per_bone: IntProperty(
        name="Max keys per bone (safety)",
        description="Limits the number of keyed samples per bone per action. 0 = unlimited. Helps prevent Blender freezing/crashing on huge rigs.",
        default=0,
        min=0,
        soft_max=20000,
    )

    action_length_mode: EnumProperty(
        name="Action length mode",
        description="How to set action.frame_end (fixes animations stopping too soon or being too long).",
        items=[
            ("DURATION", "Use animation.duration", "Use duration from JSON only."),
            ("MAX_KEYS", "Use last keyed time", "Use the last time that received keys."),
            ("MIN_OF_BOTH", "Min(duration, last keyed)", "Use whichever ends sooner."),
            ("MAX_OF_BOTH", "Max(duration, last keyed)", "Use whichever ends later."),
        ],
        default="MAX_OF_BOTH",
    )

    clamp_keys_to_duration: BoolProperty(
        name="Clamp keys to duration",
        description="If animation.duration is present, do not key anything after it (helps sequences line up).",
        default=True,
    )

    action_end_padding_frames: IntProperty(
        name="Action end padding (frames)",
        description="Adds padding to action.frame_end after length selection (0 = none). Useful if you see a 1-frame cut.",
        default=0,
        min=0,
        soft_max=10,
    )

    bone_tail_mode: EnumProperty(
        name="Bone tail mode",
        description="How to set edit bone tails. LOCAL_Y preserves the rest orientation basis (recommended for correct turret rotation axes).",
        items=[
            ("LOCAL_Y", "Local Y axis (recommended)", "Tail follows rest matrix local Y axis; preserves bone basis for animation."),
            ("TOWARD_CHILD", "Toward children (legacy)", "Tail points toward average child position; can change bone basis and alter rotation axes."),
        ],
        default="LOCAL_Y",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "exe_path")
        col.prop(self, "show_console_windows")
        col.separator()
        col.label(text="Defaults:")
        col.prop(self, "import_scale")
        col.prop(self, "import_rot_x_deg")
        col.prop(self, "flip_uv_v_default")
        col.separator()
        col.prop(self, "use_smoothing_groups_default")
        col.prop(self, "smoothing_angle_default")
        col.separator()
        col.prop(self, "apply_skinning_default")
        col.prop(self, "import_anims_default")
        col.separator()
        col.prop(self, "resample_anims")
        col.prop(self, "max_keys_per_bone")
        col.prop(self, "action_length_mode")
        col.prop(self, "clamp_keys_to_duration")
        col.prop(self, "action_end_padding_frames")
        col.separator()
        col.prop(self, "bone_tail_mode")


def _prefs(context) -> GR2ImporterPreferences:
    return context.preferences.addons[__name__].preferences


# =============================================================================
# Naming helpers
# =============================================================================

_name_sanitize_re = re.compile(r"[^A-Za-z0-9_]+")

def _sanitize_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "GR2"
    s = s.replace(" ", "_")
    s = _name_sanitize_re.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "GR2"

def _make_unique_instance_name(base: str) -> str:
    base = _sanitize_name(base)

    used: Set[str] = set()
    for c in bpy.data.collections:
        used.add(c.name)
    for o in bpy.data.objects:
        used.add(o.name)
    for a in bpy.data.actions:
        used.add(a.name)

    i = 1
    while True:
        candidate = f"{base}_{i:03d}"
        if candidate not in used and f"GR2_{candidate}" not in used:
            return candidate
        i += 1

def _unique_action_name(name: str) -> str:
    name = _sanitize_name(name)
    if bpy.data.actions.get(name) is None:
        return name
    i = 1
    while True:
        candidate = f"{name}_{i:03d}"
        if bpy.data.actions.get(candidate) is None:
            return candidate
        i += 1


# =============================================================================
# Utilities
# =============================================================================

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def _link_object(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if obj.name not in collection.objects:
        collection.objects.link(obj)

def _apply_import_transform(obj: bpy.types.Object, scale: float, rot_x_deg: float) -> None:
    rot = Matrix.Rotation(math.radians(float(rot_x_deg)), 4, 'X')
    scl = Matrix.Scale(float(scale), 4)
    obj.matrix_world = rot @ scl @ obj.matrix_world

def _quat_xyzw_to_mathutils(q: List[float]) -> Quaternion:
    # JSON: [x,y,z,w] -> mathutils: (w,x,y,z)
    if not q or len(q) != 4:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return Quaternion((w, x, y, z))

def _safe_f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _apply_smoothing_groups(obj: bpy.types.Object, enable: bool, angle_deg: float) -> None:
    """
    Blender 5: Auto Smooth is applied via object operators (not mesh.use_auto_smooth).
    Best-effort only; does not touch armature/bones.
    """
    if not enable:
        return
    if obj is None or obj.type != "MESH":
        return

    angle_rad = math.radians(float(angle_deg))

    # Prefer temp_override so we don't disturb selection/active object
    try:
        if hasattr(bpy.context, "temp_override"):
            with bpy.context.temp_override(
                active_object=obj,
                object=obj,
                selected_objects=[obj],
                selected_editable_objects=[obj],
            ):
                bpy.ops.object.shade_smooth()
                if hasattr(bpy.ops.object, "shade_auto_smooth"):
                    bpy.ops.object.shade_auto_smooth(angle=float(angle_rad))
            return
    except Exception:
        pass

    # Fallback: minimally swap selection/active and restore
    try:
        vl = bpy.context.view_layer
        prev_active = vl.objects.active
        prev_sel = [o for o in vl.objects if o.select_get()]

        for o in prev_sel:
            o.select_set(False)
        obj.select_set(True)
        vl.objects.active = obj

        bpy.ops.object.shade_smooth()
        if hasattr(bpy.ops.object, "shade_auto_smooth"):
            bpy.ops.object.shade_auto_smooth(angle=float(angle_rad))

        obj.select_set(False)
        for o in prev_sel:
            o.select_set(True)
        vl.objects.active = prev_active
    except Exception:
        pass

def _pick_first_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _get_material_name_from_index(idx_entry: Dict[str, Any], mesh_name: str, i: int) -> str:
    """
    Best-effort: infer a stable material/area name from whatever the evegr2tojson provided.
    Falls back to <mesh>_area_XX.
    """
    if not isinstance(idx_entry, dict):
        return _sanitize_name(f"{mesh_name}_area_{i:02d}")

    # Common candidates across evegr2tojsons/exports
    n = _pick_first_str(idx_entry, [
        "materialName",
        "material",
        "material_name",
        "shader",
        "shaderName",
        "effect",
        "areaName",
        "area",
        "name",
        "meshArea",
        "primitiveGroup",
        "group",
    ])
    if n:
        return _sanitize_name(n)

    # Sometimes material is nested
    mat = idx_entry.get("material", None)
    if isinstance(mat, dict):
        n2 = _pick_first_str(mat, ["name", "materialName", "shader", "shaderName"])
        if n2:
            return _sanitize_name(n2)

    return _sanitize_name(f"{mesh_name}_area_{i:02d}")

def _ensure_material(me: bpy.types.Mesh, mat_name: str) -> int:
    """
    Ensures a material datablock exists and is assigned to the mesh's material slots.
    Returns slot index.
    """
    mat_name = _sanitize_name(mat_name) or "Material"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    # Ensure in mesh slots
    for si, existing in enumerate(me.materials):
        if existing == mat:
            return si
        if existing and existing.name == mat.name:
            me.materials[si] = mat
            return si

    me.materials.append(mat)
    return len(me.materials) - 1


# =============================================================================
# Mesh import
# =============================================================================

def _make_uv_layer(mesh: bpy.types.Mesh, uv_name: str, uv_flat: List[float], flip_v: bool) -> None:
    if not uv_flat or len(uv_flat) < 2 or (len(uv_flat) % 2) != 0:
        return
    vert_uv_count = len(uv_flat) // 2
    if vert_uv_count <= 0:
        return

    uv_layer = mesh.uv_layers.new(name=uv_name)
    data = uv_layer.data

    for li, loop in enumerate(mesh.loops):
        vi = loop.vertex_index
        if 0 <= vi < vert_uv_count:
            u = float(uv_flat[2 * vi + 0])
            v = float(uv_flat[2 * vi + 1])
            if flip_v:
                v = 1.0 - v
            data[li].uv = (u, v)

def _build_faces_and_material_map(mesh_entry: Dict[str, Any], mesh_name: str) -> Tuple[List[Tuple[int, int, int]], List[int], List[str]]:
    """
    Builds:
      faces: list of triangle tuples
      face_group_ids: per-face group index (0..g-1) matching areas/submeshes
      group_names: group index -> inferred name
    """
    faces: List[Tuple[int, int, int]] = []
    face_group_ids: List[int] = []
    group_names: List[str] = []

    indices_list = mesh_entry.get("indices", [])
    if not isinstance(indices_list, list) or not indices_list:
        # fallback: no groups
        return faces, face_group_ids, group_names

    for gi, idx in enumerate(indices_list):
        if not isinstance(idx, dict):
            continue
        flat = idx.get("faces", None)
        if not isinstance(flat, list) or not flat:
            continue
        if (len(flat) % 3) != 0:
            continue

        group_names.append(_get_material_name_from_index(idx, mesh_name=mesh_name, i=gi))

        tri_count = len(flat) // 3
        for t in range(tri_count):
            a = int(flat[t * 3 + 0])
            b = int(flat[t * 3 + 1])
            c = int(flat[t * 3 + 2])
            faces.append((a, b, c))
            face_group_ids.append(gi)

    return faces, face_group_ids, group_names

def import_meshes(
    gr2: Dict[str, Any],
    collection: bpy.types.Collection,
    instance_name: str,
    scale: float,
    rot_x_deg: float,
    use_smoothing_groups: bool,
    smoothing_angle_deg: float,
    flip_uv_v: bool,
) -> List[Tuple[bpy.types.Object, Dict[str, Any]]]:
    created: List[Tuple[bpy.types.Object, Dict[str, Any]]] = []

    meshes = gr2.get("meshes", [])
    if not isinstance(meshes, list) or not meshes:
        return created

    for m in meshes:
        if not isinstance(m, dict):
            continue

        raw_mesh_name = m.get("name", "GR2Mesh")
        mesh_name = _sanitize_name(raw_mesh_name)
        obj_name = f"{instance_name}_{mesh_name}"

        v = m.get("vertex", {})
        if not isinstance(v, dict):
            continue

        pos = v.get("position", [])
        if not isinstance(pos, list) or len(pos) < 3 or (len(pos) % 3) != 0:
            continue

        vert_count = len(pos) // 3
        verts = [(float(pos[i*3+0]), float(pos[i*3+1]), float(pos[i*3+2])) for i in range(vert_count)]

        faces, face_group_ids, group_names = _build_faces_and_material_map(m, mesh_name=mesh_name)

        # Fallback if no grouped faces came through
        if not faces:
            # try legacy path if exporter uses a single combined list (rare, but keep compatible)
            indices_list = m.get("indices", [])
            faces_flat: List[int] = []
            if isinstance(indices_list, list):
                for idx in indices_list:
                    if not isinstance(idx, dict):
                        continue
                    ff = idx.get("faces", None)
                    if isinstance(ff, list):
                        faces_flat.extend(int(x) for x in ff)
            if not faces_flat or (len(faces_flat) % 3) != 0:
                continue
            faces = [(int(faces_flat[i*3+0]), int(faces_flat[i*3+1]), int(faces_flat[i*3+2]))
                     for i in range(len(faces_flat)//3)]
            face_group_ids = [0] * len(faces)
            group_names = [_sanitize_name(f"{mesh_name}_area_00")]

        me = bpy.data.meshes.new(name=obj_name)
        me.from_pydata(verts, [], faces)
        me.update(calc_edges=True)

        # UVs
        uv0 = v.get("texcoord0", None)
        if isinstance(uv0, list):
            _make_uv_layer(me, "UV0", uv0, flip_v=flip_uv_v)
        uv1 = v.get("texcoord1", None)
        if isinstance(uv1, list):
            _make_uv_layer(me, "UV1", uv1, flip_v=flip_uv_v)

        # Materials / mesh areas: create slots and assign polygon.material_index
        # Ensure we have a stable slot per group index.
        group_to_slot: Dict[int, int] = {}
        for gi, gname in enumerate(group_names):
            # If name is empty for some reason, still stable per group index
            if not gname:
                gname = _sanitize_name(f"{mesh_name}_area_{gi:02d}")
            slot = _ensure_material(me, gname)
            group_to_slot[gi] = slot

        # Assign per polygon in creation order (matches faces list order)
        if face_group_ids and len(me.polygons) == len(face_group_ids):
            for pi, p in enumerate(me.polygons):
                gi = int(face_group_ids[pi])
                p.material_index = int(group_to_slot.get(gi, 0))

        obj = bpy.data.objects.new(name=obj_name, object_data=me)
        _link_object(obj, collection)
        _apply_import_transform(obj, scale=scale, rot_x_deg=rot_x_deg)

        _apply_smoothing_groups(obj, enable=use_smoothing_groups, angle_deg=smoothing_angle_deg)

        created.append((obj, m))

    return created


# =============================================================================
# Skeleton helpers (rest matrices)
# =============================================================================

def _mat_from_scale_shear_3x3(ss: List[float]) -> Matrix:
    if not isinstance(ss, list) or len(ss) != 9:
        return Matrix.Identity(3)
    return Matrix((
        (float(ss[0]), float(ss[1]), float(ss[2])),
        (float(ss[3]), float(ss[4]), float(ss[5])),
        (float(ss[6]), float(ss[7]), float(ss[8])),
    ))

def _local_matrix_from_bone_json(b: Dict[str, Any]) -> Matrix:
    pos = b.get("position", [0.0, 0.0, 0.0])
    ori = b.get("orientation", [0.0, 0.0, 0.0, 1.0])
    ss  = b.get("scaleShear", None)

    t = Vector((_safe_f(pos[0], 0.0), _safe_f(pos[1], 0.0), _safe_f(pos[2], 0.0))) if isinstance(pos, list) and len(pos) == 3 else Vector((0.0, 0.0, 0.0))
    q = _quat_xyzw_to_mathutils(ori) if isinstance(ori, list) and len(ori) == 4 else Quaternion((1.0, 0.0, 0.0, 0.0))
    R = q.to_matrix().to_4x4()

    SS3 = _mat_from_scale_shear_3x3(ss) if isinstance(ss, list) and len(ss) == 9 else Matrix.Identity(3)
    SS = SS3.to_4x4()

    return Matrix.Translation(t) @ R @ SS

def _compute_rest_matrices(bones: List[Dict[str, Any]]) -> Tuple[List[Matrix], List[Matrix]]:
    n = len(bones)
    rest_local = [Matrix.Identity(4) for _ in range(n)]
    rest_world = [Matrix.Identity(4) for _ in range(n)]

    for i, b in enumerate(bones):
        rest_local[i] = _local_matrix_from_bone_json(b)

    for i, b in enumerate(bones):
        parent = int(b.get("parentIndex", -1))
        if 0 <= parent < n:
            rest_world[i] = rest_world[parent] @ rest_local[i]
        else:
            rest_world[i] = rest_local[i]

    return rest_local, rest_world


# =============================================================================
# Armature import
# =============================================================================

def import_armature(
    gr2: Dict[str, Any],
    collection: bpy.types.Collection,
    instance_name: str,
    scale: float,
    rot_x_deg: float,
    bone_tail_mode: str,
) -> Tuple[Optional[bpy.types.Object], Dict[str, Any]]:
    models = gr2.get("models", [])
    if not isinstance(models, list) or not models:
        return None, {}

    skel = models[0].get("skeleton", None)
    if not isinstance(skel, dict):
        return None, {}

    bones = skel.get("bones", [])
    if not isinstance(bones, list) or not bones:
        return None, {}

    arm_name = f"{instance_name}_Armature"
    arm_data = bpy.data.armatures.new(arm_name)
    arm_obj = bpy.data.objects.new(arm_name, arm_data)
    _link_object(arm_obj, collection)

    _apply_import_transform(arm_obj, scale=scale, rot_x_deg=rot_x_deg)

    rest_local, rest_world = _compute_rest_matrices(bones)

    name_to_index: Dict[str, int] = {}
    for i, b in enumerate(bones):
        bn = (b.get("name") or f"Bone_{i}")
        if isinstance(bn, str):
            name_to_index[bn] = i

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones: List[bpy.types.EditBone] = []
    for i, b in enumerate(bones):
        bn = (b.get("name") or f"Bone_{i}")
        eb = arm_data.edit_bones.new(bn)
        edit_bones.append(eb)

    for i, b in enumerate(bones):
        parent = int(b.get("parentIndex", -1))
        if 0 <= parent < len(edit_bones):
            edit_bones[i].parent = edit_bones[parent]

    children: List[List[int]] = [[] for _ in range(len(bones))]
    for i, b in enumerate(bones):
        p = int(b.get("parentIndex", -1))
        if 0 <= p < len(bones):
            children[p].append(i)

    for i, eb in enumerate(edit_bones):
        M = rest_world[i].copy()
        eb.matrix = M

        head = Vector(eb.head)

        # Length heuristic: if there are children, use distance to average child; else a small default.
        length = 0.05
        if children[i]:
            avg = Vector((0.0, 0.0, 0.0))
            for ci in children[i]:
                avg += Vector(rest_world[ci].to_translation())
            avg /= max(1, len(children[i]))
            dist = (avg - head).length
            length = max(dist, 0.05)

        if bone_tail_mode == "TOWARD_CHILD" and children[i]:
            # Legacy behavior (can change bone basis!)
            avg = Vector((0.0, 0.0, 0.0))
            for ci in children[i]:
                avg += Vector(rest_world[ci].to_translation())
            avg /= max(1, len(children[i]))
            direction = (avg - head)
            if direction.length < 1e-6:
                direction = (M.to_3x3() @ Vector((0.0, 0.1, 0.0)))
            eb.tail = head + direction.normalized() * length
        else:
            # Recommended: preserve rest basis by aligning tail to rest local Y axis
            y_axis = (M.to_3x3() @ Vector((0.0, 1.0, 0.0)))
            if y_axis.length < 1e-6:
                y_axis = Vector((0.0, 0.1, 0.0))
            eb.tail = head + y_axis.normalized() * length

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_obj.select_set(False)

    return arm_obj, {
        "bones_json": bones,
        "name_to_index": name_to_index,
        "rest_local": rest_local,
        "rest_world": rest_world,
    }


# =============================================================================
# Skinning
# =============================================================================

def _ensure_armature_modifier(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object) -> None:
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE" and mod.object == arm_obj:
            return
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

def _get_binding_bone_name(mesh_entry: Dict[str, Any], binding_index: int) -> Optional[str]:
    binds = mesh_entry.get("boneBindings", [])
    if not isinstance(binds, list):
        return None
    if 0 <= binding_index < len(binds):
        b = binds[binding_index]
        if isinstance(b, dict):
            n = b.get("name", None)
            return n if isinstance(n, str) and n else None
    return None

def _as_int_index(v) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return 0

def _parse_weights4(raw4: List[Any]) -> List[float]:
    w = [float(x) for x in raw4]
    maxw = max(w) if w else 0.0
    if maxw > 1.0001:
        scale = 255.0 if maxw <= 255.0 else 65535.0
        w = [wi / scale for wi in w]
    w = [0.0 if wi < 0.0 else (1.0 if wi > 1.0 else wi) for wi in w]
    s = w[0] + w[1] + w[2] + w[3]
    if s > 0.0:
        w = [wi / s for wi in w]
    else:
        w = [1.0, 0.0, 0.0, 0.0]
    return w

def apply_skinning(mesh_obj: bpy.types.Object, mesh_entry: Dict[str, Any], arm_obj: bpy.types.Object) -> None:
    if mesh_obj.type != "MESH" or arm_obj is None or arm_obj.type != "ARMATURE":
        return

    v = mesh_entry.get("vertex", {})
    if not isinstance(v, dict):
        return

    pos = v.get("position", [])
    if not isinstance(pos, list) or (len(pos) % 3) != 0:
        return
    vert_count = len(pos) // 3
    if vert_count <= 0:
        return

    blend_i = v.get("blendIndice", [])
    blend_w = v.get("blendWeight", [])
    if not isinstance(blend_i, list) or len(blend_i) < vert_count * 4:
        return

    has_weights = isinstance(blend_w, list) and len(blend_w) >= vert_count * 4
    if has_weights:
        sample = blend_w[:min(len(blend_w), 4096)]
        try:
            if all(float(x) == 0.0 for x in sample):
                has_weights = False
        except Exception:
            pass

    _ensure_armature_modifier(mesh_obj, arm_obj)

    vgroups: Dict[str, bpy.types.VertexGroup] = {}
    def get_vg(name: str) -> bpy.types.VertexGroup:
        vg = vgroups.get(name)
        if vg is None:
            vg = mesh_obj.vertex_groups.get(name)
            if vg is None:
                vg = mesh_obj.vertex_groups.new(name=name)
            vgroups[name] = vg
        return vg

    arm_bones = arm_obj.data.bones

    for vi in range(vert_count):
        idx_base = vi * 4
        weights = _parse_weights4(blend_w[idx_base:idx_base+4]) if has_weights else [1.0, 0.0, 0.0, 0.0]

        for j in range(4):
            w = float(weights[j])
            if w <= 0.0:
                continue

            binding_index = _as_int_index(blend_i[idx_base + j])
            bone_name = _get_binding_bone_name(mesh_entry, binding_index)
            if not bone_name:
                continue
            if bone_name not in arm_bones:
                continue

            vg = get_vg(bone_name)
            vg.add([vi], w, "REPLACE")


# =============================================================================
# Curve decoding
# =============================================================================

def _u32_to_f32_bits(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

def _one_over_knot_scale_from_trunc(trunc_u16: int) -> float:
    u = (int(trunc_u16) & 0xFFFF) << 16
    f = _u32_to_f32_bits(u)
    return float(f) if f != 0.0 else 1.0

def _get_knot_count(curve: Dict[str, Any]) -> int:
    kc = curve.get("knotCount", None)
    if isinstance(kc, int) and kc >= 0:
        return kc

    fmt = int(curve.get("format", -1))

    if fmt == 1:
        knots = curve.get("knots", [])
        return len(knots) if isinstance(knots, list) else 0

    if fmt == 0:
        dim = curve.get("dimension", 0)
        controls = curve.get("controls", [])
        if isinstance(dim, int) and dim > 0 and isinstance(controls, list):
            return len(controls) // dim
        return 0

    knc = curve.get("knotsControls", [])
    if not isinstance(knc, list):
        if fmt == 2:
            return 0
        if fmt in (3, 4, 5):
            return 1
        return 0

    total = len(knc)
    if fmt in (8, 9, 10, 11, 13, 15):
        return total // 4
    if fmt in (12, 14, 16, 17, 18):
        return total // 2
    if fmt in (6, 7):
        cso = curve.get("controlScaleOffsets", [])
        if isinstance(cso, list) and len(cso) % 2 == 0:
            dim = len(cso) // 2
            denom = dim + 1
            return total // denom if denom > 0 else 0

    return 0


_SCALE_TABLE = [
    1.4142135, 0.70710677, 0.35355338, 0.35355338,
    0.35355338, 0.17677669, 0.17677669, 0.17677669,
    -1.4142135, -0.70710677, -0.35355338, -0.35355338,
    -0.35355338, -0.17677669, -0.17677669, -0.17677669,
]
_OFFSET_TABLE = [
    -0.70710677, -0.35355338, -0.53033006, -0.17677669,
    0.17677669, -0.17677669, -0.088388346, 0.0,
    0.70710677, 0.35355338, 0.53033006, 0.17677669,
    -0.17677669, 0.17677669, 0.088388346, -0.0,
]

def _create_quat_15u(a: int, b: int, c: int, scales: List[float], offsets: List[float]) -> List[float]:
    swizzle1 = ((b & 0x8000) >> 14) | (c >> 15)
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3

    dataA = (a & 0x7FFF) * scales[swizzle2] + offsets[swizzle2]
    dataB = (b & 0x7FFF) * scales[swizzle3] + offsets[swizzle3]
    dataC = (c & 0x7FFF) * scales[swizzle4] + offsets[swizzle4]

    d2 = 1.0 - (dataA*dataA + dataB*dataB + dataC*dataC)
    dataD = math.sqrt(max(d2, 0.0))
    if (a & 0x8000) != 0:
        dataD = -dataD

    quat = [0.0, 0.0, 0.0, 0.0]
    quat[swizzle2] = dataA
    quat[swizzle3] = dataB
    quat[swizzle4] = dataC
    quat[swizzle1] = dataD
    return quat

def _create_quat_7u(a: int, b: int, c: int, scales: List[float], offsets: List[float]) -> List[float]:
    swizzle1 = ((b & 0x80) >> 6) | ((c & 0x80) >> 7)
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3

    dataA = (a & 0x7F) * scales[swizzle2] + offsets[swizzle2]
    dataB = (b & 0x7F) * scales[swizzle3] + offsets[swizzle3]
    dataC = (c & 0x7F) * scales[swizzle4] + offsets[swizzle4]

    d2 = 1.0 - (dataA*dataA + dataB*dataB + dataC*dataC)
    dataD = math.sqrt(max(d2, 0.0))
    if (a & 0x80) != 0:
        dataD = -dataD

    quat = [0.0, 0.0, 0.0, 0.0]
    quat[swizzle2] = dataA
    quat[swizzle3] = dataB
    quat[swizzle4] = dataC
    quat[swizzle1] = dataD
    return quat

def decode_curve(curve: Dict[str, Any], expected_dim: int, time_step: float) -> Tuple[List[float], List[List[float]]]:
    fmt = int(curve.get("format", -1))
    k = _get_knot_count(curve)

    if fmt == 2:
        return ([], [])

    if fmt in (3, 4, 5):
        controls = curve.get("controls", [])
        if isinstance(controls, list) and len(controls) >= expected_dim:
            return ([0.0], [[float(controls[j]) for j in range(expected_dim)]])
        return ([0.0], [[0.0]*expected_dim])

    if fmt == 0:
        controls = curve.get("controls", [])
        if not isinstance(controls, list) or expected_dim <= 0:
            return ([], [])
        count = len(controls) // expected_dim
        times = [i * float(time_step) for i in range(count)]
        vals = []
        for i in range(count):
            base = i * expected_dim
            vals.append([float(controls[base + j]) for j in range(expected_dim)])
        return (times, vals)

    if fmt == 1:
        knots = curve.get("knots", [])
        controls = curve.get("controls", [])
        if not (isinstance(knots, list) and isinstance(controls, list)):
            return ([], [])
        times = [float(t) for t in knots]
        vals = []
        for i in range(len(times)):
            base = i * expected_dim
            if base + expected_dim <= len(controls):
                vals.append([float(controls[base + j]) for j in range(expected_dim)])
            else:
                break
        return (times, vals)

    knc = curve.get("knotsControls", [])
    if not isinstance(knc, list) or k <= 0:
        return ([], [])

    if fmt == 8 and expected_dim == 4:
        selector = int(curve.get("scaleOffsetTableEntries", 0))
        one_over = float(curve.get("oneOverKnotScale", 1.0)) or 1.0

        scales = [0.0]*4
        offsets = [0.0]*4
        mul = 0.000030518509
        for i in range(4):
            idx = (selector >> (i*4)) & 0x0F
            scales[i] = _SCALE_TABLE[idx] * mul
            offsets[i] = _OFFSET_TABLE[idx]

        times = [float(knc[i]) / one_over for i in range(k)]
        vals = []
        base = k
        for i in range(k):
            a = int(knc[base + i*3 + 0])
            b = int(knc[base + i*3 + 1])
            c = int(knc[base + i*3 + 2])
            q = _create_quat_15u(a, b, c, scales, offsets)
            l = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) or 1.0
            vals.append([q[0]/l, q[1]/l, q[2]/l, q[3]/l])
        return (times, vals)

    if fmt == 9 and expected_dim == 4:
        selector = int(curve.get("scaleOffsetTableEntries", 0))
        one_over = float(curve.get("oneOverKnotScale", 1.0)) or 1.0

        scales = [0.0]*4
        offsets = [0.0]*4
        mul = 0.0078740157
        for i in range(4):
            idx = (selector >> (i*4)) & 0x0F
            scales[i] = _SCALE_TABLE[idx] * mul
            offsets[i] = _OFFSET_TABLE[idx]

        times = [float(knc[i]) / one_over for i in range(k)]
        vals = []
        base = k
        for i in range(k):
            a = int(knc[base + i*3 + 0])
            b = int(knc[base + i*3 + 1])
            c = int(knc[base + i*3 + 2])
            q = _create_quat_7u(a, b, c, scales, offsets)
            l = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) or 1.0
            vals.append([q[0]/l, q[1]/l, q[2]/l, q[3]/l])
        return (times, vals)

    if fmt in (10, 11) and expected_dim == 3:
        trunc = int(curve.get("oneOverKnotScaleTrunc", 0))
        one_over = _one_over_knot_scale_from_trunc(trunc)
        scales = curve.get("controlScales", [1.0, 1.0, 1.0])
        offsets = curve.get("controlOffsets", [0.0, 0.0, 0.0])

        times = [float(knc[i]) / one_over for i in range(k)]
        vals = []
        base = k
        for i in range(k):
            c0 = int(knc[base + i*3 + 0])
            c1 = int(knc[base + i*3 + 1])
            c2 = int(knc[base + i*3 + 2])
            vals.append([
                c0 * float(scales[0]) + float(offsets[0]),
                c1 * float(scales[1]) + float(offsets[1]),
                c2 * float(scales[2]) + float(offsets[2]),
            ])
        return (times, vals)

    if fmt in (17, 18) and expected_dim == 3:
        trunc = int(curve.get("oneOverKnotScaleTrunc", 0))
        one_over = _one_over_knot_scale_from_trunc(trunc)
        scales = curve.get("controlScales", [1.0, 1.0, 1.0])
        offsets = curve.get("controlOffsets", [0.0, 0.0, 0.0])

        times = [float(knc[i]) / one_over for i in range(k)]
        vals = []
        base = k
        for i in range(k):
            s = int(knc[base + i])
            vals.append([
                s * float(scales[0]) + float(offsets[0]),
                s * float(scales[1]) + float(offsets[1]),
                s * float(scales[2]) + float(offsets[2]),
            ])
        return (times, vals)

    return ([], [])


# =============================================================================
# Animation import (resample + slerp)
# =============================================================================

def _set_scene_fps_from_time_step(time_step: float) -> None:
    if time_step <= 0.0:
        return
    desired = 1.0 / float(time_step)
    fps_int = int(round(desired))
    fps_int = max(1, min(240, fps_int))

    scene = bpy.context.scene
    scene.render.fps = fps_int
    scene.render.fps_base = float(fps_int) / float(desired)

def _effective_fps(scene: bpy.types.Scene) -> float:
    base = float(scene.render.fps_base) if scene.render.fps_base else 1.0
    return float(scene.render.fps) / base

def _time_to_frame(t: float, fps: float) -> float:
    return float(t) * float(fps)

def _mat_from_scale_shear_3x3(ss: List[float]) -> Matrix:
    if not isinstance(ss, list) or len(ss) != 9:
        return Matrix.Identity(3)
    return Matrix((
        (float(ss[0]), float(ss[1]), float(ss[2])),
        (float(ss[3]), float(ss[4]), float(ss[5])),
        (float(ss[6]), float(ss[7]), float(ss[8])),
    ))

def _mat_from_anim_components(pos3: List[float], quat4_xyzw: List[float], ss9: List[float]) -> Matrix:
    T = Matrix.Translation(Vector((pos3[0], pos3[1], pos3[2])))
    R = _quat_xyzw_to_mathutils(quat4_xyzw).to_matrix().to_4x4()
    SS3 = _mat_from_scale_shear_3x3(ss9) if isinstance(ss9, list) and len(ss9) == 9 else Matrix.Identity(3)
    SS = SS3.to_4x4()
    return T @ R @ SS

def _get_max_time_from_tracks(pos_t, rot_t, ss_t) -> float:
    mt = 0.0
    if pos_t:
        mt = max(mt, float(pos_t[-1]))
    if rot_t:
        mt = max(mt, float(rot_t[-1]))
    if ss_t:
        mt = max(mt, float(ss_t[-1]))
    return mt

def _ensure_quat_continuity_xyzw(values: List[List[float]]) -> List[List[float]]:
    if not values:
        return values
    out = [values[0][:]]
    prev = _quat_xyzw_to_mathutils(values[0]).normalized()
    for i in range(1, len(values)):
        q = _quat_xyzw_to_mathutils(values[i]).normalized()
        if prev.dot(q) < 0.0:
            q = Quaternion((-q.w, -q.x, -q.y, -q.z))
        out.append([q.x, q.y, q.z, q.w])
        prev = q
    q0 = _quat_xyzw_to_mathutils(out[0]).normalized()
    out[0] = [q0.x, q0.y, q0.z, q0.w]
    return out

def _lerp_list(a: List[float], b: List[float], f: float) -> List[float]:
    inv = 1.0 - f
    return [a[i] * inv + b[i] * f for i in range(len(a))]

def _find_segment(times: List[float], t: float) -> Tuple[int, float]:
    if not times:
        return (0, 0.0)
    if t <= times[0]:
        return (0, 0.0)
    last = len(times) - 1
    if t >= times[last]:
        return (last, 0.0)

    for i in range(last):
        t0 = times[i]
        t1 = times[i + 1]
        if t0 <= t <= t1:
            denom = (t1 - t0)
            f = (t - t0) / denom if denom > 0.0 else 0.0
            return (i, max(0.0, min(1.0, f)))
    return (last, 0.0)

def _eval_vec_curve(times: List[float], vals: List[List[float]], t: float, default: List[float]) -> List[float]:
    if not times or not vals:
        return default
    if len(times) == 1 or len(vals) == 1:
        v = vals[0]
        return [float(v[i]) for i in range(len(default))]

    i0, f = _find_segment(times, t)
    if i0 >= len(vals) - 1:
        v = vals[-1]
        return [float(v[i]) for i in range(len(default))]
    a = vals[i0]
    b = vals[i0 + 1]
    a2 = [float(a[i]) for i in range(len(default))]
    b2 = [float(b[i]) for i in range(len(default))]
    return _lerp_list(a2, b2, f)

def _eval_quat_curve_xyzw(times: List[float], vals_xyzw: List[List[float]], t: float, default_xyzw: List[float]) -> List[float]:
    if not times or not vals_xyzw:
        return default_xyzw
    if len(times) == 1 or len(vals_xyzw) == 1:
        q = _quat_xyzw_to_mathutils(vals_xyzw[0]).normalized()
        return [q.x, q.y, q.z, q.w]

    i0, f = _find_segment(times, t)
    if i0 >= len(vals_xyzw) - 1:
        q = _quat_xyzw_to_mathutils(vals_xyzw[-1]).normalized()
        return [q.x, q.y, q.z, q.w]

    qa = _quat_xyzw_to_mathutils(vals_xyzw[i0]).normalized()
    qb = _quat_xyzw_to_mathutils(vals_xyzw[i0 + 1]).normalized()
    if qa.dot(qb) < 0.0:
        qb = Quaternion((-qb.w, -qb.x, -qb.y, -qb.z))

    q = qa.slerp(qb, f).normalized()
    return [q.x, q.y, q.z, q.w]

def _choose_action_end_frame(length_mode: str, dur_end: float, key_end: float) -> float:
    dur_end = float(dur_end) if dur_end else 0.0
    key_end = float(key_end) if key_end else 0.0

    if length_mode == "DURATION":
        return dur_end
    if length_mode == "MAX_KEYS":
        return key_end
    if length_mode == "MIN_OF_BOTH":
        candidates = [x for x in (dur_end, key_end) if x > 0.0]
        return min(candidates) if candidates else 0.0
    return max(dur_end, key_end)

def import_animations(
    gr2: Dict[str, Any],
    arm_obj: bpy.types.Object,
    arm_info: Dict[str, Any],
    instance_name: str,
    resample: bool,
    max_keys_per_bone: int,
    action_length_mode: str,
    clamp_keys_to_duration: bool,
    action_end_padding_frames: int,
) -> List[bpy.types.Action]:
    actions: List[bpy.types.Action] = []
    anims = gr2.get("animations", [])
    if not isinstance(anims, list) or not anims:
        return actions
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return actions

    bones_json: List[Dict[str, Any]] = arm_info.get("bones_json", [])
    name_to_index: Dict[str, int] = arm_info.get("name_to_index", {})
    rest_local: List[Matrix] = arm_info.get("rest_local", [])

    if not bones_json or not name_to_index or not rest_local:
        return actions

    arm_obj.animation_data_create()
    scene = bpy.context.scene

    for anim in anims:
        if not isinstance(anim, dict):
            continue

        anim_name_raw = anim.get("name", "Action") or "Action"
        anim_name = _sanitize_name(anim_name_raw)
        full_action_name = _unique_action_name(f"{instance_name}_{anim_name}")

        duration = float(anim.get("duration", 0.0) or 0.0)
        time_step = float(anim.get("timeStep", 1.0/30.0) or (1.0/30.0))
        oversampling = float(anim.get("oversampling", 1.0) or 1.0)
        oversampling = max(1.0, oversampling)

        _set_scene_fps_from_time_step(time_step)
        fps = _effective_fps(scene)

        action = bpy.data.actions.new(name=full_action_name)
        actions.append(action)
        arm_obj.animation_data.action = action

        max_frame_keys = 0.0

        bone_tracks: Dict[str, Dict[str, Any]] = {}
        track_groups = anim.get("trackGroups", [])
        if isinstance(track_groups, list):
            for tg in track_groups:
                if not isinstance(tg, dict):
                    continue
                tracks = tg.get("transformTracks", [])
                if not isinstance(tracks, list):
                    continue
                for tr in tracks:
                    if not isinstance(tr, dict):
                        continue
                    bn = tr.get("name", None)
                    if isinstance(bn, str) and bn:
                        bone_tracks[bn] = tr

        for bn, pb in arm_obj.pose.bones.items():
            if bn not in name_to_index:
                continue
            bidx = name_to_index[bn]
            restM = rest_local[bidx]
            restMinv = restM.inverted_safe()

            tr = bone_tracks.get(bn, None)

            rest_b = bones_json[bidx]
            def_pos = rest_b.get("position", [0.0, 0.0, 0.0])
            def_rot = rest_b.get("orientation", [0.0, 0.0, 0.0, 1.0])
            def_ss  = rest_b.get("scaleShear", [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0])

            def_pos3 = [float(def_pos[i]) for i in range(3)] if isinstance(def_pos, list) and len(def_pos) == 3 else [0.0, 0.0, 0.0]
            def_rot4 = [float(def_rot[i]) for i in range(4)] if isinstance(def_rot, list) and len(def_rot) == 4 else [0.0, 0.0, 0.0, 1.0]
            def_ss9  = [float(def_ss[i])  for i in range(9)] if isinstance(def_ss, list) and len(def_ss) == 9 else [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]

            pos_times: List[float] = []
            pos_vals: List[List[float]] = []
            rot_times: List[float] = []
            rot_vals: List[List[float]] = []
            ss_times: List[float] = []
            ss_vals: List[List[float]] = []

            if isinstance(tr, dict):
                pc = tr.get("position", {})
                oc = tr.get("orientation", {})
                sc = tr.get("scaleShear", {})

                if isinstance(pc, dict):
                    pos_times, pos_vals = decode_curve(pc, expected_dim=3, time_step=time_step)
                if isinstance(oc, dict):
                    rot_times, rot_vals = decode_curve(oc, expected_dim=4, time_step=time_step)
                if isinstance(sc, dict):
                    ss_times, ss_vals = decode_curve(sc, expected_dim=9, time_step=time_step)

            if rot_times and rot_vals:
                rot_vals = _ensure_quat_continuity_xyzw(rot_vals)

            # Determine sampling duration
            if duration > 0.0:
                local_duration = duration
            else:
                local_duration = _get_max_time_from_tracks(pos_times, rot_times, ss_times)

            if resample:
                dt = float(time_step) / float(oversampling)
                dt = max(1e-6, dt)

                steps = int(math.floor((local_duration / dt) + 0.5)) if local_duration > 0.0 else 0
                sample_times = [i * dt for i in range(max(1, steps + 1))]
                if local_duration > 0.0 and (not sample_times or abs(sample_times[-1] - local_duration) > 1e-6):
                    sample_times.append(local_duration)
            else:
                s: Set[float] = set()
                for t in pos_times:
                    s.add(float(t))
                for t in rot_times:
                    s.add(float(t))
                for t in ss_times:
                    s.add(float(t))
                sample_times = sorted(s) if s else [0.0]

            # Clamp to animation.duration if requested and duration is present
            if clamp_keys_to_duration and duration > 0.0:
                d = float(duration)
                sample_times = [t for t in sample_times if t <= d + 1e-8]
                if not sample_times:
                    sample_times = [0.0]
                if abs(sample_times[-1] - d) > 1e-6:
                    sample_times.append(d)

            # Safety: limit number of keyed samples per bone.
            if max_keys_per_bone and max_keys_per_bone > 0 and len(sample_times) > max_keys_per_bone:
                sample_times = sample_times[:max_keys_per_bone]

            pb.rotation_mode = "QUATERNION"

            for t in sample_times:
                pos3 = _eval_vec_curve(pos_times, pos_vals, t, def_pos3)
                rot4 = _eval_quat_curve_xyzw(rot_times, rot_vals, t, def_rot4)
                ss9  = _eval_vec_curve(ss_times,  ss_vals,  t, def_ss9)

                anim_local = _mat_from_anim_components(pos3, rot4, ss9)
                delta = restMinv @ anim_local

                loc = delta.to_translation()
                rot = delta.to_quaternion()
                scl = delta.to_scale()

                frame = _time_to_frame(t, fps)
                if frame > max_frame_keys:
                    max_frame_keys = frame

                pb.location = loc
                pb.rotation_quaternion = rot
                pb.scale = scl

                pb.keyframe_insert(data_path="location", frame=frame, group=bn)
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bn)
                pb.keyframe_insert(data_path="scale", frame=frame, group=bn)

        dur_end = _time_to_frame(duration, fps) if duration > 0.0 else 0.0
        end = _choose_action_end_frame(action_length_mode, dur_end=dur_end, key_end=max_frame_keys)

        # If clamping is enabled and duration exists, never let end exceed duration end (+ padding)
        if clamp_keys_to_duration and duration > 0.0:
            end = min(float(end), float(dur_end))

        end = float(end) + float(action_end_padding_frames or 0)

        action.frame_start = 0.0
        action.frame_end = max(1.0, end) if end > 0.0 else max(1.0, float(max_frame_keys))

        try:
            scene.frame_end = max(scene.frame_end, int(math.ceil(action.frame_end)))
        except Exception:
            pass

    arm_obj.animation_data.action = None
    return actions


# =============================================================================
# High-level import
# =============================================================================

def import_gr2_json(
    gr2: Dict[str, Any],
    base_name: str,
    apply_skinning_flag: bool,
    import_anims_flag: bool,
    scale: float,
    rot_x_deg: float,
    flip_uv_v: bool,
    use_smoothing_groups: bool,
    smoothing_angle_deg: float,
    resample_anims: bool,
    max_keys_per_bone: int,
    action_length_mode: str,
    clamp_keys_to_duration: bool,
    action_end_padding_frames: int,
    bone_tail_mode: str,
) -> Dict[str, Any]:
    instance_name = _make_unique_instance_name(base_name)
    col = _ensure_collection(f"GR2_{instance_name}")

    mesh_pairs = import_meshes(
        gr2, col,
        instance_name=instance_name,
        scale=scale,
        rot_x_deg=rot_x_deg,
        use_smoothing_groups=use_smoothing_groups,
        smoothing_angle_deg=smoothing_angle_deg,
        flip_uv_v=flip_uv_v,
    )
    arm_obj, arm_info = import_armature(
        gr2, col, instance_name=instance_name, scale=scale, rot_x_deg=rot_x_deg, bone_tail_mode=bone_tail_mode
    )

    if apply_skinning_flag and arm_obj is not None:
        for mesh_obj, mesh_entry in mesh_pairs:
            apply_skinning(mesh_obj, mesh_entry, arm_obj)

    actions: List[bpy.types.Action] = []
    if import_anims_flag and arm_obj is not None:
        actions = import_animations(
            gr2,
            arm_obj,
            arm_info,
            instance_name=instance_name,
            resample=resample_anims,
            max_keys_per_bone=max_keys_per_bone,
            action_length_mode=action_length_mode,
            clamp_keys_to_duration=clamp_keys_to_duration,
            action_end_padding_frames=action_end_padding_frames,
        )

    return {
        "instance_name": instance_name,
        "collection": col,
        "meshes": [o for (o, _) in mesh_pairs],
        "armature": arm_obj,
        "actions": actions,
    }


# =============================================================================
# Operators
# =============================================================================

class IMPORT_OT_gr2_via_exe(Operator, ImportHelper):
    bl_idname = "import_scene.gr2_via_exe"
    bl_label = "Import GR2 (via EXE)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".gr2"
    filter_glob: StringProperty(default="*.gr2", options={"HIDDEN"})

    apply_skinning: BoolProperty(
        name="Apply Skinning",
        description="Create vertex groups + armature modifier using blendIndice/blendWeight.",
        default=True,
    )

    import_animations: BoolProperty(
        name="Import Animations",
        description="Create Actions from animations[] onto the armature.",
        default=True,
    )

    flip_uv_v: BoolProperty(
        name="Flip UV V (Y)",
        description="Flips V (Y) coordinate for UV layers on import.",
        default=True,
    )

    use_smoothing_groups: BoolProperty(
        name="Smoothing groups",
        description="Applies Smooth shading + Auto Smooth (angle) to imported meshes.",
        default=True,
    )

    smoothing_angle: FloatProperty(
        name="Smoothing angle (degrees)",
        description="Angle threshold for Auto Smooth (0–180). Higher = smoother.",
        default=180.0,
        min=0.0,
        max=180.0,
    )

    def invoke(self, context, event):
        prefs = _prefs(context)
        self.apply_skinning = bool(prefs.apply_skinning_default)
        self.import_animations = bool(prefs.import_anims_default)
        self.flip_uv_v = bool(prefs.flip_uv_v_default)
        self.use_smoothing_groups = bool(prefs.use_smoothing_groups_default)
        self.smoothing_angle = float(prefs.smoothing_angle_default)
        return super().invoke(context, event)

    def execute(self, context):
        prefs = _prefs(context)
        exe = bpy.path.abspath(prefs.exe_path) if prefs.exe_path else ""
        if not exe or not os.path.isfile(exe):
            self.report({"ERROR"}, "Set a valid evegr2tojson EXE path in Add-on Preferences.")
            return {"CANCELLED"}

        gr2_path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(gr2_path):
            self.report({"ERROR"}, "GR2 file not found.")
            return {"CANCELLED"}

        out_json = os.path.splitext(gr2_path)[0] + ".gr2_json"

        creationflags = 0
        if os.name == "nt" and not prefs.show_console_windows:
            creationflags = 0x08000000  # CREATE_NO_WINDOW

        try:
            subprocess.run([exe, gr2_path], check=True, creationflags=creationflags)
        except subprocess.CalledProcessError as e:
            self.report({"ERROR"}, f"evegr2tojson EXE failed: {e}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to run evegr2tojson EXE: {e}")
            return {"CANCELLED"}

        if not os.path.isfile(out_json):
            self.report({"ERROR"}, f"Expected output JSON not found: {out_json}")
            return {"CANCELLED"}

        try:
            gr2 = _load_json(out_json)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load JSON: {e}")
            return {"CANCELLED"}

        base_name = os.path.basename(os.path.splitext(gr2_path)[0])
        result = import_gr2_json(
            gr2,
            base_name,
            apply_skinning_flag=bool(self.apply_skinning),
            import_anims_flag=bool(self.import_animations),
            scale=float(prefs.import_scale),
            rot_x_deg=float(prefs.import_rot_x_deg),
            flip_uv_v=bool(self.flip_uv_v),
            use_smoothing_groups=bool(self.use_smoothing_groups),
            smoothing_angle_deg=float(self.smoothing_angle),
            resample_anims=bool(prefs.resample_anims),
            max_keys_per_bone=int(prefs.max_keys_per_bone),
            action_length_mode=str(prefs.action_length_mode),
            clamp_keys_to_duration=bool(prefs.clamp_keys_to_duration),
            action_end_padding_frames=int(prefs.action_end_padding_frames),
            bone_tail_mode=str(prefs.bone_tail_mode),
        )
        self.report({"INFO"}, f"Imported: {result.get('instance_name')}")
        return {"FINISHED"}


class IMPORT_OT_gr2_json(Operator, ImportHelper):
    bl_idname = "import_scene.gr2_json"
    bl_label = "Import GR2 JSON"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".gr2_json"
    filter_glob: StringProperty(default="*.gr2_json", options={"HIDDEN"})

    apply_skinning: BoolProperty(
        name="Apply Skinning",
        description="Create vertex groups + armature modifier using blendIndice/blendWeight.",
        default=True,
    )

    import_animations: BoolProperty(
        name="Import Animations",
        description="Create Actions from animations[] onto the armature.",
        default=True,
    )

    flip_uv_v: BoolProperty(
        name="Flip UV V (Y)",
        description="Flips V (Y) coordinate for UV layers on import.",
        default=True,
    )

    use_smoothing_groups: BoolProperty(
        name="Smoothing groups",
        description="Applies Smooth shading + Auto Smooth (angle) to imported meshes.",
        default=True,
    )

    smoothing_angle: FloatProperty(
        name="Smoothing angle (degrees)",
        description="Angle threshold for Auto Smooth (0–180). Higher = smoother.",
        default=180.0,
        min=0.0,
        max=180.0,
    )

    def invoke(self, context, event):
        prefs = _prefs(context)
        self.apply_skinning = bool(prefs.apply_skinning_default)
        self.import_animations = bool(prefs.import_anims_default)
        self.flip_uv_v = bool(prefs.flip_uv_v_default)
        self.use_smoothing_groups = bool(prefs.use_smoothing_groups_default)
        self.smoothing_angle = float(prefs.smoothing_angle_default)
        return super().invoke(context, event)

    def execute(self, context):
        prefs = _prefs(context)
        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({"ERROR"}, "JSON file not found.")
            return {"CANCELLED"}

        try:
            gr2 = _load_json(path)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load JSON: {e}")
            return {"CANCELLED"}

        base_name = os.path.basename(os.path.splitext(path)[0])
        result = import_gr2_json(
            gr2,
            base_name,
            apply_skinning_flag=bool(self.apply_skinning),
            import_anims_flag=bool(self.import_animations),
            scale=float(prefs.import_scale),
            rot_x_deg=float(prefs.import_rot_x_deg),
            flip_uv_v=bool(self.flip_uv_v),
            use_smoothing_groups=bool(self.use_smoothing_groups),
            smoothing_angle_deg=float(self.smoothing_angle),
            resample_anims=bool(prefs.resample_anims),
            max_keys_per_bone=int(prefs.max_keys_per_bone),
            action_length_mode=str(prefs.action_length_mode),
            clamp_keys_to_duration=bool(prefs.clamp_keys_to_duration),
            action_end_padding_frames=int(prefs.action_end_padding_frames),
            bone_tail_mode=str(prefs.bone_tail_mode),
        )
        self.report({"INFO"}, f"Imported: {result.get('instance_name')}")
        return {"FINISHED"}


# =============================================================================
# Menu registration
# =============================================================================

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_gr2_via_exe.bl_idname, text="GR2 (via EXE)")
    self.layout.operator(IMPORT_OT_gr2_json.bl_idname, text="GR2 JSON (.gr2_json)")

classes = (
    GR2ImporterPreferences,
    IMPORT_OT_gr2_via_exe,
    IMPORT_OT_gr2_json,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()