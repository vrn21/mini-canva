"""Layout aesthetics scorer — overlap, alignment, margin, spacing."""

from __future__ import annotations

from engine.canvas import Canvas

_MIN_MARGIN = 20
_ALIGN_TOLERANCE = 10


class AestheticsScorer:
    """Scores visual layout quality of elements on a canvas."""

    def score(self, canvas: Canvas) -> float:
        """Overall aesthetics score."""

        if canvas.element_count == 0:
            return 0.0

        if canvas.element_count == 1:
            return self._margin_score(canvas)

        scores = [
            self._overlap_score(canvas),
            self._alignment_score(canvas),
            self._margin_score(canvas),
            self._spacing_score(canvas),
        ]
        return sum(scores) / len(scores)

    def _overlap_score(self, canvas: Canvas) -> float:
        """1.0 = no overlaps, 0.0 = severe overlaps."""

        elements = canvas.get_all_elements()
        total_element_area = sum(element.area for element in elements)
        if total_element_area == 0:
            return 1.0

        total_overlap_area = sum(area for _, _, area in canvas.get_overlapping_pairs())
        score = 1.0 - (total_overlap_area / total_element_area)
        return max(0.0, score)

    def _alignment_score(self, canvas: Canvas) -> float:
        """Score how many elements share common alignment axes."""

        elements = canvas.get_all_elements()
        n = len(elements)
        if n < 2:
            return 1.0

        center_xs = [element.center[0] for element in elements]
        center_ys = [element.center[1] for element in elements]
        left_edges = [float(element.x) for element in elements]

        best = 0.0
        for values in (center_xs, center_ys, left_edges):
            best = max(best, _max_cluster_fraction(values, _ALIGN_TOLERANCE, n))

        return best

    def _margin_score(self, canvas: Canvas) -> float:
        """Fraction of elements with adequate margins from canvas edges."""

        elements = canvas.get_all_elements()
        canvas_width = canvas.config.width
        canvas_height = canvas.config.height

        respecting = 0
        for element in elements:
            left, top, right, bottom = element.bounds
            if (
                left >= _MIN_MARGIN
                and top >= _MIN_MARGIN
                and right <= canvas_width - _MIN_MARGIN
                and bottom <= canvas_height - _MIN_MARGIN
            ):
                respecting += 1

        return respecting / len(elements)

    def _spacing_score(self, canvas: Canvas) -> float:
        """Regularity of vertical gaps between elements."""

        elements = canvas.get_all_elements()
        if len(elements) < 3:
            return 1.0

        centers_y = sorted(element.center[1] for element in elements)
        gaps = [centers_y[i + 1] - centers_y[i] for i in range(len(centers_y) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return 1.0

        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        stddev = variance ** 0.5
        normalized_stddev = stddev / mean_gap
        return max(0.0, 1.0 - normalized_stddev)


def _max_cluster_fraction(values: list[float], tolerance: float, n: int) -> float:
    """Find the largest cluster of values within ±tolerance."""

    if n == 0:
        return 0.0

    best_count = 1
    for i in range(n):
        count = sum(1 for j in range(n) if abs(values[i] - values[j]) <= tolerance)
        best_count = max(best_count, count)

    return best_count / n
