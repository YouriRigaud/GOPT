"""
OnlineBPH: Online Bin Packing Heuristic for 3D Container Loading Problem.

The module accepts the same box dimensions used by the environment
(`(length, width, height)` tuples or `Box`-like objects with `size_x`,
`size_y`, `size_z`) so it can be plugged into the existing packing setup
without maintaining a separate item model.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Tolerance for floating-point comparisons
_EPS = 1e-9


def _item_dims(item: Any) -> Tuple[float, float, float]:
    """Return `(length, width, height)` for tuples or Box-like objects."""
    if hasattr(item, "size_x") and hasattr(item, "size_y") and hasattr(item, "size_z"):
        return float(item.size_x), float(item.size_y), float(item.size_z)
    if hasattr(item, "length") and hasattr(item, "width") and hasattr(item, "height"):
        return float(item.length), float(item.width), float(item.height)
    length, width, height = item[:3]
    return float(length), float(width), float(height)


def _item_weight(item: Any) -> float:
    """Return item weight — index 3 if tuple/list, else .weight attribute, default 1.0."""
    if hasattr(item, "weight"):
        return float(item.weight)
    try:
        return float(item[3])
    except (IndexError, TypeError):
        return 1.0


def _item_fragility(item: Any) -> bool:
    """Return True if item is fragile — index 4 if tuple/list, else .fragility attribute."""
    if hasattr(item, "fragility"):
        return bool(item.fragility)
    try:
        return bool(item[4])
    except (IndexError, TypeError):
        return False


def _item_volume(item: Any) -> float:
    length, width, height = _item_dims(item)
    return length * width * height


def _item_rotations(item: Any) -> List[Tuple[float, float, float]]:
    """Return all 6 axis-aligned rotations of an item."""
    l, w, h = _item_dims(item)
    seen = set()
    result = []
    for dims in [(l, w, h), (w, l, h), (l, h, w), (w, h, l), (h, l, w), (h, w, l)]:
        if dims not in seen:
            seen.add(dims)
            result.append(dims)
    return result


def _dims_volume(dims: Tuple[float, float, float]) -> float:
    return dims[0] * dims[1] * dims[2]


@dataclass
class EMS:
    """
    Empty Maximal Space (EMS) representation.

    Attributes:
        min_corner: (x_min, y_min, z_min) - deepest-bottom-left
        max_corner: (x_max, y_max, z_max) - highest-top-right
    """
    min_corner: Tuple[float, float, float]
    max_corner: Tuple[float, float, float]

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return (dx, dy, dz) of the EMS."""
        return (
            self.max_corner[0] - self.min_corner[0],
            self.max_corner[1] - self.min_corner[1],
            self.max_corner[2] - self.min_corner[2],
        )

    @property
    def volume(self) -> float:
        dx, dy, dz = self.dimensions
        return dx * dy * dz

    def priority_key(self) -> Tuple[float, float, float]:
        """
        Sort priority: primary = x_min (deepest),
                       secondary = z_min (lowest),
                       tertiary = y_min (leftmost).
        """
        x_min, y_min, z_min = self.min_corner
        return (x_min, z_min, y_min)

    def fits(self, item_dims: Tuple[float, float, float]) -> bool:
        """Check if item dimensions fit inside this EMS."""
        l_item, w_item, h_item = item_dims
        dx, dy, dz = self.dimensions
        return l_item <= dx + _EPS and w_item <= dy + _EPS and h_item <= dz + _EPS


class OnlineBPHContainer:
    """
    Container managed by the OnlineBPH algorithm.

    Coordinate system: x = depth (door at x = length),
                       y = width, z = height.
    """

    def __init__(self, container_id: int, length: float, width: float, height: float):
        self.container_id = container_id
        self.length = length   # x dimension — door at x = length
        self.width  = width    # y dimension
        self.height = height   # z dimension

        # Each entry: (original_dims, rotation_dims, position, weight, fragility)
        self.placed_items: List[
            Tuple[
                Tuple[float, float, float],  # original dims
                Tuple[float, float, float],  # rotation dims
                Tuple[float, float, float],  # position (x, y, z)
                float,                       # weight
                bool,                        # fragile
            ]
        ] = []

        # Initial EMS covers the whole container
        self.ems_list: List[EMS] = [
            EMS(min_corner=(0.0, 0.0, 0.0),
                max_corner=(length, width, height))
        ]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_sorted_ems(self, ke: int) -> List[EMS]:
        """Return the top *ke* EMSs sorted by priority (x, z, y)."""
        return sorted(self.ems_list, key=lambda e: e.priority_key())[:ke]

    def place_item(
        self,
        item_dims: Tuple[float, float, float],
        rotation_dims: Tuple[float, float, float],
        ems: EMS,
        weight: float = 1.0,
        fragile: bool = False,
    ) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """
        Place item at the min_corner of *ems* and update the EMS list.

        Returns (success, position) or (False, None).
        """
        if not ems.fits(rotation_dims):
            return False, None

        item_pos = ems.min_corner
        self.placed_items.append((item_dims, rotation_dims, item_pos, weight, fragile))
        self._update_ems_difference(rotation_dims, item_pos)
        return True, item_pos

    def get_volume_utilization(self) -> float:
        """Volume utilisation ratio in [0, 1]."""
        placed_volume = sum(
            _dims_volume(entry[1])   # entry[1] = rotation_dims
            for entry in self.placed_items
        )
        container_volume = self.length * self.width * self.height
        return placed_volume / container_volume if container_volume > 0 else 0.0

    # ------------------------------------------------------------------
    # Fragility constraint helpers
    # ------------------------------------------------------------------

    def has_fragile_item_at(self, pos: Tuple[float, float, float],
                             dims: Tuple[float, float, float]) -> bool:
        """
        Return True if any fragile item already placed in this container
        would be directly below the footprint of a new item placed at *pos*
        with *dims* (l, w, h).

        A fragile item is "below" the new item if their footprints overlap
        in x-y AND the fragile item's top face (z + h_frag) equals the new
        item's bottom face (z_new).  We allow a small tolerance.
        """
        nx, ny, nz = pos
        nl, nw, _  = dims

        for entry in self.placed_items:
            _, rot_dims, fpos, _, fragile = entry
            if not fragile:
                continue
            fx, fy, fz = fpos
            fl, fw, fh = rot_dims
            # Check z: fragile item's top must touch new item's bottom
            if abs((fz + fh) - nz) > _EPS:
                continue
            # Check x-y footprint overlap
            x_overlap = fx < nx + nl - _EPS and fx + fl > nx + _EPS
            y_overlap = fy < ny + nw - _EPS and fy + fw > ny + _EPS
            if x_overlap and y_overlap:
                return True
        return False

    # ------------------------------------------------------------------
    # Centre-of-gravity helpers
    # ------------------------------------------------------------------

    def compute_cog(self) -> Tuple[float, float, float]:
        """
        Compute the current centre of gravity of all placed items.
        Uses item weight (entry[3]) and geometric centre of each item.
        Returns (cx, cy, cz).  Returns container centre if no items placed.
        """
        if not self.placed_items:
            return (self.length / 2, self.width / 2, self.height / 2)

        total_weight = 0.0
        wx_sum = wy_sum = wz_sum = 0.0
        for entry in self.placed_items:
            _, rot_dims, pos, weight, _ = entry
            l, w, h = rot_dims
            cx = pos[0] + l / 2
            cy = pos[1] + w / 2
            cz = pos[2] + h / 2
            wx_sum += weight * cx
            wy_sum += weight * cy
            wz_sum += weight * cz
            total_weight += weight

        return (wx_sum / total_weight,
                wy_sum / total_weight,
                wz_sum / total_weight)

    def cog_distance_to_center(self) -> float:
        """
        Euclidean distance of the current CoG from the geometric centre
        of the container floor plan (x-y plane only, ignoring z).
        Using x-y only is standard practice for load balance.
        """
        cx, cy, _ = self.compute_cog()
        return float(np.sqrt((cx - self.length / 2) ** 2 +
                              (cy - self.width  / 2) ** 2))

    def simulated_cog_distance(
        self,
        rotation_dims: Tuple[float, float, float],
        pos: Tuple[float, float, float],
        weight: float,
    ) -> float:
        """
        Compute CoG distance to container centre *as if* an item with
        *rotation_dims*, *pos*, *weight* were added — without mutating state.
        Used as a tiebreaker in placement selection.
        """
        total_weight = weight
        l, w, h = rotation_dims
        wx_sum = weight * (pos[0] + l / 2)
        wy_sum = weight * (pos[1] + w / 2)

        for entry in self.placed_items:
            _, rot_dims, epos, ew, _ = entry
            el, ew_dim, _ = rot_dims
            wx_sum += ew * (epos[0] + el / 2)
            wy_sum += ew * (epos[1] + ew_dim / 2)
            total_weight += ew

        if total_weight < _EPS:
            return 0.0

        cx = wx_sum / total_weight
        cy = wy_sum / total_weight
        return float(np.sqrt((cx - self.length / 2) ** 2 +
                              (cy - self.width  / 2) ** 2))

    def is_empty(self) -> bool:
        return len(self.placed_items) == 0

    # ------------------------------------------------------------------
    # EMS update — difference process (Lai & Chan 1997)
    # ------------------------------------------------------------------

    def _update_ems_difference(
        self,
        item_dims: Tuple[float, float, float],
        item_pos: Tuple[float, float, float],
    ) -> None:
        """
        Split all EMSs that overlap with the newly placed item into up to
        6 sub-EMSs (one per face of the item), then remove non-maximal ones.
        """
        l, w, h = item_dims
        ix, iy, iz = item_pos
        ix2, iy2, iz2 = ix + l, iy + w, iz + h

        new_ems_list: List[EMS] = []

        for ems in self.ems_list:
            if not self._ems_overlaps_item(ems, item_pos, item_dims):
                new_ems_list.append(ems)
                continue

            # Generate up to 6 sub-EMSs
            x0, y0, z0 = ems.min_corner
            x1, y1, z1 = ems.max_corner

            candidates = [
                # Beyond item in +x direction
                EMS((ix2, y0, z0), (x1,  y1, z1)) if ix2 < x1 - _EPS else None,
                # Beyond item in +y direction
                EMS((x0, iy2, z0), (x1,  y1, z1)) if iy2 < y1 - _EPS else None,
                # Beyond item in +z direction
                EMS((x0, y0, iz2), (x1,  y1, z1)) if iz2 < z1 - _EPS else None,
                # Before item in -x direction
                EMS((x0, y0, z0), (ix,  y1, z1))  if ix  > x0 + _EPS else None,
                # Before item in -y direction
                EMS((x0, y0, z0), (x1,  iy, z1))  if iy  > y0 + _EPS else None,
                # Before item in -z direction
                EMS((x0, y0, z0), (x1,  y1, iz))  if iz  > z0 + _EPS else None,
            ]

            for cand in candidates:
                if cand is not None and self._is_valid_ems(cand):
                    new_ems_list.append(cand)

        self.ems_list = new_ems_list
        self._remove_non_maximal_ems()

    # ------------------------------------------------------------------
    # EMS utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _ems_overlaps_item(
        ems: EMS,
        item_pos: Tuple[float, float, float],
        item_dims: Tuple[float, float, float],
    ) -> bool:
        x0, y0, z0 = ems.min_corner
        x1, y1, z1 = ems.max_corner
        ix, iy, iz = item_pos
        l,  w,  h  = item_dims
        return (
            x0 < ix + l - _EPS and x1 > ix + _EPS and
            y0 < iy + w - _EPS and y1 > iy + _EPS and
            z0 < iz + h - _EPS and z1 > iz + _EPS
        )

    @staticmethod
    def _is_valid_ems(ems: EMS) -> bool:
        return (
            ems.max_corner[0] - ems.min_corner[0] > _EPS and
            ems.max_corner[1] - ems.min_corner[1] > _EPS and
            ems.max_corner[2] - ems.min_corner[2] > _EPS
        )

    def _remove_non_maximal_ems(self) -> None:
        """Keep only EMSs that are not fully contained in another EMS."""
        if len(self.ems_list) <= 1:
            return

        maximal: List[EMS] = []
        for i, e1 in enumerate(self.ems_list):
            dominated = False
            for j, e2 in enumerate(self.ems_list):
                if i == j:
                    continue
                if self._inscribed_in(e1, e2) and not self._ems_equal(e1, e2):
                    dominated = True
                    break
            if not dominated:
                maximal.append(e1)

        # Safety: always keep at least one EMS
        self.ems_list = maximal if maximal else [self.ems_list[0]]

    @staticmethod
    def _inscribed_in(small: EMS, large: EMS) -> bool:
        return (
            small.min_corner[0] >= large.min_corner[0] - _EPS and
            small.min_corner[1] >= large.min_corner[1] - _EPS and
            small.min_corner[2] >= large.min_corner[2] - _EPS and
            small.max_corner[0] <= large.max_corner[0] + _EPS and
            small.max_corner[1] <= large.max_corner[1] + _EPS and
            small.max_corner[2] <= large.max_corner[2] + _EPS
        )

    @staticmethod
    def _ems_equal(e1: EMS, e2: EMS) -> bool:
        return (
            all(abs(e1.min_corner[i] - e2.min_corner[i]) < _EPS for i in range(3)) and
            all(abs(e1.max_corner[i] - e2.max_corner[i]) < _EPS for i in range(3))
        )


# ---------------------------------------------------------------------------
# Infinite container sequence helper
# ---------------------------------------------------------------------------

def _infinite_containers(
    dims: Tuple[float, float, float]
) -> Iterator[Tuple[float, float, float]]:
    """Yield identical container dimensions indefinitely."""
    while True:
        yield dims


class OnlineBPH:
    """
    Online Bin Packing Heuristic (OnlineBPH) — Ha et al. (2017).

    Parameters
    ----------
    kb : int
        Look-ahead window (number of buffered items considered together).
        Use kb=1 for strict online (one item at a time).
    ke : int
        Number of top-priority EMSs to examine per open container.
    """

    def __init__(self, kb: int = 1, ke: int = 1):
        self.kb = kb
        self.ke = ke
        self.open_containers:   List[OnlineBPHContainer] = []
        self.closed_containers: List[OnlineBPHContainer] = []
        self._next_id = 0
        self._buffer: List[Any] = []   # look-ahead buffer for kb > 1

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def pack(
        self,
        item: Any,
        container_sequence: Any,  # Sequence or Iterator of (L, W, H)
    ) -> bool:
        """
        Pack a single item online.

        *container_sequence* may be a list/tuple **or** an iterator (including
        the infinite generator returned by `_infinite_containers`).

        Returns True if the item was successfully placed.
        """
        # --- kb > 1: buffer items and pack when buffer is full -----------
        if self.kb > 1:
            self._buffer.append(item)
            if len(self._buffer) < self.kb:
                return True  # defer — item will be packed with the batch
            return self._pack_buffer(container_sequence)

        # --- kb == 1: strict online, one item at a time ------------------
        return self._pack_single(item, container_sequence)

    def flush(self, container_sequence: Any) -> bool:
        """
        Pack any items remaining in the look-ahead buffer (call at the end
        of the item stream when kb > 1).
        """
        if not self._buffer:
            return True
        return self._pack_buffer(container_sequence)

    def run(
        self,
        items: Sequence[Any],
        container_sequence: Any,
    ) -> Dict:
        """
        Run OnlineBPH on a full sequence of items.

        *container_sequence* can be:
          - A list/tuple of (L, W, H) tuples (finite).
          - An iterator / generator (e.g. ``_infinite_containers(dims)``).

        Returns a result dictionary.
        """
        packed_count  = 0
        failed_count  = 0

        for item in items:
            if self.pack(item, container_sequence):
                packed_count += 1
            else:
                failed_count += 1

        # Flush the look-ahead buffer
        if self.kb > 1:
            self.flush(container_sequence)

        # Close open containers
        self.closed_containers.extend(self.open_containers)
        self.open_containers = []

        # --- Statistics ---------------------------------------------------
        total_item_volume = sum(_item_volume(it) for it in items)
        total_container_volume = sum(
            c.length * c.width * c.height
            for c in self.closed_containers
        )
        utilization = (
            total_item_volume / total_container_volume * 100
            if total_container_volume > 0 else 0.0
        )

        return {
            "num_items":              len(items),
            "packed_items":           packed_count,
            "failed_items":           failed_count,
            "num_containers":         len(self.closed_containers),
            "total_item_volume":      total_item_volume,
            "total_container_volume": total_container_volume,
            "utilization_percent":    utilization,
            "containers":             self.closed_containers,
        }

    # ------------------------------------------------------------------
    # Internal packing logic
    # ------------------------------------------------------------------

    def _pack_single(self, item: Any, container_sequence: Any) -> bool:
        """Place one item immediately."""
        best = self._find_best_placement([item], self.open_containers)

        if best is not None:
            container, _, rotation_dims, ems = best
            success, _ = container.place_item(
                _item_dims(item), rotation_dims, ems,
                weight=_item_weight(item),
                fragile=_item_fragility(item),
            )
            return success

        # Open a new container
        return self._open_and_place(item, container_sequence)

    def _pack_buffer(self, container_sequence: Any) -> bool:
        """
        Pack all items currently in self._buffer (kb > 1 mode).

        Items are packed one by one in the order that maximises the
        fill ratio of the best available placement, consistent with the
        paper's description of the look-ahead mechanism.
        """
        items = list(self._buffer)
        self._buffer.clear()
        all_ok = True

        while items:
            # Find the best (item, container, rotation, ems) across all buffered items
            best_global = None
            best_fill   = 0.0
            best_margin = float("inf")
            best_item   = None

            for item in items:
                result = self._find_best_placement([item], self.open_containers)
                if result is None:
                    continue
                _, _, rotation_dims, ems = result
                fill   = _item_volume(item) / ems.volume
                margin = sum(ems.dimensions[i] - rotation_dims[i] for i in range(3))
                if (fill > best_fill or
                        (abs(fill - best_fill) < _EPS and margin < best_margin)):
                    best_global = result
                    best_fill   = fill
                    best_margin = margin
                    best_item   = item

            if best_global is not None and best_item is not None:
                container, _, rotation_dims, ems = best_global
                container.place_item(
                    _item_dims(best_item), rotation_dims, ems,
                    weight=_item_weight(best_item),
                    fragile=_item_fragility(best_item),
                )
                items.remove(best_item)
            else:
                # No item fits in any open container — open a new one for the first item
                item = items.pop(0)
                if not self._open_and_place(item, container_sequence):
                    all_ok = False

        return all_ok

    def _find_best_placement(
        self,
        items: List[Any],
        containers: List[OnlineBPHContainer],
    ) -> Optional[Tuple]:
        """
        Find the best tetrad (container, item, rotation, ems) for the given
        list of items across all open containers.

        Feasibility constraints
        -----------------------
        1. Blocking check   : EMS must reach the container door (x_max == L).
        2. Fragility check  : new item must NOT be placed on top of a fragile item.

        Scoring (lexicographic)
        -----------------------
        1. Fill ratio       : item_volume / ems_volume  — higher is better.
        2. Margin           : sum of leftover space per axis — lower is better.
        3. CoG distance     : simulated distance of CoG to container centre
                              after adding this item — lower is better.

        Returns (container, item, rotation_dims, ems) or None.
        """
        best_container  = None
        best_item       = None
        best_rotation   = None
        best_ems        = None
        # Score tuple: (fill_ratio, -margin, -cog_dist)  — maximise
        best_score: Tuple[float, float, float] = (-1.0, -float("inf"), -float("inf"))

        for container in containers:
            sorted_ems = container.get_sorted_ems(self.ke)

            for ems in sorted_ems:
                # --- 1. Blocking check (Section 3.1) --------------------
                if abs(ems.max_corner[0] - container.length) > _EPS:
                    continue

                pos = ems.min_corner   # item placed at EMS min_corner

                for item in items:
                    weight = _item_weight(item)

                    for rotation_dims in _item_rotations(item):
                        if not ems.fits(rotation_dims):
                            continue

                        # --- 2. Fragility constraint --------------------
                        # Reject if any fragile item is directly below
                        if container.has_fragile_item_at(pos, rotation_dims):
                            continue

                        # --- 3. Scoring ---------------------------------
                        fill_ratio = _item_volume(item) / ems.volume
                        margin     = sum(
                            ems.dimensions[i] - rotation_dims[i] for i in range(3)
                        )
                        cog_dist   = container.simulated_cog_distance(
                            rotation_dims, pos, weight
                        )
                        score = (fill_ratio, -margin, -cog_dist)

                        if score > best_score:
                            best_score      = score
                            best_container  = container
                            best_item       = item
                            best_rotation   = rotation_dims
                            best_ems        = ems

        if best_container is None:
            return None
        return best_container, best_item, best_rotation, best_ems

    def _open_and_place(self, item: Any, container_sequence: Any) -> bool:
        """
        Open the next available container (FindNewSuitableContainer) and
        place *item* inside it with the best-fitting rotation.

        Raises StopIteration / returns False when the sequence is exhausted.
        """
        # Support both lists and iterators
        try:
            if hasattr(container_sequence, "__next__"):
                container_dims = next(container_sequence)
            else:
                container_dims = container_sequence[self._next_id]
        except (StopIteration, IndexError):
            return False  # No more containers available

        L, W, H = container_dims

        # Find the rotation of the item that fits and maximises fill ratio
        best_rotation = None
        best_fill     = 0.0
        for rotation_dims in _item_rotations(item):
            l, w, h = rotation_dims
            if l <= L + _EPS and w <= W + _EPS and h <= H + _EPS:
                fill = _item_volume(item) / (L * W * H)
                if fill > best_fill:
                    best_fill     = fill
                    best_rotation = rotation_dims

        if best_rotation is None:
            # Item does not fit in any orientation — skip container and retry
            return False

        new_container = OnlineBPHContainer(self._next_id, L, W, H)
        self._next_id += 1
        self.open_containers.append(new_container)

        ems = new_container.ems_list[0]  # full-container EMS
        success, _ = new_container.place_item(
            _item_dims(item), best_rotation, ems,
            weight=_item_weight(item),
            fragile=_item_fragility(item),
        )
        return success