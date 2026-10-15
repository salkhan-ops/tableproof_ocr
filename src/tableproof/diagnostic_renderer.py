from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tableproof.models import TableGrid


def render_diagnostic_png(table_grid: TableGrid, output_path: Path | str) -> None:
    output_path = Path(output_path)
    if table_grid.bounding_box is None:
        raise ValueError("Cannot render diagnostics without a table bounding box")

    scale = 2.0
    padding = 32
    bbox = table_grid.bounding_box
    width = max(320, int(bbox.width * scale) + padding * 2)
    height = max(220, int(bbox.height * scale) + padding * 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def tx(x: float) -> int:
        return int((x - bbox.x0) * scale) + padding

    def ty(y: float) -> int:
        return int((y - bbox.y0) * scale) + padding

    for row_gap in table_grid.metrics.get("per_row_word_gaps", []):
        for word in row_gap.get("gaps", []):
            if word["row_index"] >= len(table_grid.rows):
                continue
            left = tx(word["x0"])
            right = tx(word["x1"])
            center = tx(word["center_x"])
            row = table_grid.rows[word["row_index"]]
            if row.bounding_box is None:
                continue
            top = ty(row.bounding_box.y0) + 3
            bottom = ty(row.bounding_box.y1) - 3
            draw.rectangle((left, top, right, bottom), outline=(130, 130, 130), width=1)
            draw.line((center, top, center, bottom), fill=(160, 160, 160), width=1)

    for row in table_grid.rows:
        if row.bounding_box is None:
            continue
        y0 = ty(row.bounding_box.y0)
        y1 = ty(row.bounding_box.y1)
        draw.line((padding, y0, width - padding, y0), fill=(40, 130, 220), width=2)
        draw.line((padding, y1, width - padding, y1), fill=(40, 130, 220), width=1)
        draw.text((4, max(0, y0 - 6)), f"r{row.row_index}", fill=(20, 80, 160), font=font)

    for anchor in table_grid.metrics.get("candidate_column_anchors", []):
        x = tx(anchor["position"])
        for y in range(padding // 2, height - padding // 2, 10):
            draw.line((x, y, x, min(y + 5, height - padding // 2)), fill=(140, 80, 190), width=1)
        draw.text((x + 3, height - 22), f"a{anchor['index']}", fill=(100, 50, 150), font=font)

    for column_index, boundary in enumerate(table_grid.column_boundaries):
        x = tx(boundary)
        draw.line((x, padding // 2, x, height - padding // 2), fill=(220, 90, 40), width=2)
        if column_index < table_grid.column_count:
            draw.text((x + 3, 6), f"c{column_index}", fill=(170, 60, 20), font=font)

    for row in table_grid.rows:
        for cell in row.cells:
            if cell.bounding_box is None:
                left = tx(table_grid.column_boundaries[cell.column_index])
                right = tx(table_grid.column_boundaries[cell.column_index + 1])
                if row.bounding_box is None:
                    continue
                top = ty(row.bounding_box.y0)
                bottom = ty(row.bounding_box.y1)
                outline = (180, 180, 180)
            else:
                left = tx(cell.bounding_box.x0)
                top = ty(cell.bounding_box.y0)
                right = tx(cell.bounding_box.x1)
                bottom = ty(cell.bounding_box.y1)
                outline = (40, 170, 90) if cell.confidence >= 0.75 else (210, 70, 70)
            draw.rectangle((left, top, right, bottom), outline=outline, width=2)
            label = f"{cell.row_index},{cell.column_index}"
            draw.text((left + 2, top + 2), label, fill=(30, 30, 30), font=font)

    for span in table_grid.merged_ranges():
        coords = span["covered_coordinates"]
        top_left = coords[0]
        bottom_right = coords[-1]
        first_row = table_grid.rows[top_left[0]]
        last_row = table_grid.rows[bottom_right[0]]
        left = tx(table_grid.column_boundaries[top_left[1]])
        right = tx(table_grid.column_boundaries[bottom_right[1] + 1])
        top = ty(first_row.bounding_box.y0) if first_row.bounding_box else padding
        bottom = ty(last_row.bounding_box.y1) if last_row.bounding_box else height - padding
        draw.rectangle((left, top, right, bottom), outline=(150, 60, 200), width=3)
        label = "MIXED-SPAN" if span["row_span"] > 1 and span["column_span"] > 1 else ("ROW-SPAN" if span["row_span"] > 1 else "COLUMN-SPAN")
        draw.text((left + 3, max(0, top - 12)), label, fill=(120, 40, 170), font=font)
        for covered_row, covered_col in map(tuple, span["covered_coordinates"][1:]):
            if covered_row >= len(table_grid.rows):
                continue
            covered_line = table_grid.rows[covered_row]
            if covered_col + 1 >= len(table_grid.column_boundaries) or covered_line.bounding_box is None:
                continue
            c_left = tx(table_grid.column_boundaries[covered_col])
            c_right = tx(table_grid.column_boundaries[covered_col + 1])
            c_top = ty(covered_line.bounding_box.y0)
            c_bottom = ty(covered_line.bounding_box.y1)
            draw.rectangle((c_left, c_top, c_right, c_bottom), outline=(80, 80, 80), width=1)
            draw.text((c_left + 2, c_top + 2), "COVERED", fill=(80, 80, 80), font=font)

    for node in table_grid.metrics.get("header_hierarchy", {}).get("nodes", []):
        draw.text((width - 120, 12 + 10 * int(node.get("hierarchy_level", 0))), "PARENT" if node.get("child_candidate_ids") else "CHILD", fill=(80, 40, 160), font=font)

    for coord in table_grid.metrics.get("genuine_empty_coordinate_map", []):
        row_index, column_index = coord
        if row_index >= len(table_grid.rows) or column_index + 1 >= len(table_grid.column_boundaries):
            continue
        row = table_grid.rows[row_index]
        if row.bounding_box is None:
            continue
        left = tx(table_grid.column_boundaries[column_index])
        right = tx(table_grid.column_boundaries[column_index + 1])
        top = ty(row.bounding_box.y0)
        bottom = ty(row.bounding_box.y1)
        draw.rectangle((left, top, right, bottom), outline=(30, 150, 150), width=1)
        draw.text((left + 2, bottom - 10), "EMPTY", fill=(20, 120, 120), font=font)

    for block in table_grid.metrics.get("excluded_text_blocks", []):
        block_bbox = block.get("bounding_box")
        if not block_bbox:
            continue
        left = tx(block_bbox["x0"])
        top = ty(block_bbox["y0"])
        right = tx(block_bbox["x1"])
        bottom = ty(block_bbox["y1"])
        draw.rectangle((left, top, right, bottom), outline=(210, 40, 40), width=3)
        draw.text((left + 3, top + 3), block.get("classification", "EXCLUDED"), fill=(170, 20, 20), font=font)

    raw_line_boxes = {
        line.get("line_index"): line.get("bounding_box")
        for line in table_grid.metrics.get("raw_visual_lines", [])
        if line.get("bounding_box")
    }
    for decision in table_grid.metrics.get("selected_decisions", []):
        candidate = decision.get("selected_candidate", {})
        source_indices = candidate.get("source_line_indices", [])
        if not source_indices:
            continue
        source_box = raw_line_boxes.get(source_indices[0])
        target_row = candidate.get("target_row_index")
        target_col = candidate.get("target_column_index")
        if source_box is None or target_row is None or target_col is None or target_row >= len(table_grid.rows):
            continue
        target_cells = table_grid.rows[target_row].cells
        if target_col >= len(target_cells) or target_cells[target_col].bounding_box is None:
            continue
        target_box = target_cells[target_col].bounding_box
        start = (tx((source_box["x0"] + source_box["x1"]) / 2), ty((source_box["y0"] + source_box["y1"]) / 2))
        end = (tx((target_box.x0 + target_box.x1) / 2), ty((target_box.y0 + target_box.y1) / 2))
        draw.line((start, end), fill=(20, 120, 120), width=2)
        draw.text((start[0] + 3, start[1] + 3), "cont", fill=(20, 100, 100), font=font)

    for unresolved in table_grid.metrics.get("unresolved_lines", []):
        line_box = raw_line_boxes.get(unresolved.get("line_index"))
        if not line_box:
            continue
        left = tx(line_box["x0"])
        top = ty(line_box["y0"])
        right = tx(line_box["x1"])
        bottom = ty(line_box["y1"])
        draw.rectangle((left, top, right, bottom), outline=(230, 160, 20), width=3)
        draw.text((left + 3, top + 3), "unresolved", fill=(180, 120, 20), font=font)

    for unresolved in table_grid.metrics.get("unresolved_spans", []):
        row_index = unresolved.get("anchor_row")
        column_index = unresolved.get("anchor_column")
        if row_index is None or column_index is None or row_index >= len(table_grid.rows):
            continue
        row = table_grid.rows[row_index]
        if row.bounding_box is None or column_index + 1 >= len(table_grid.column_boundaries):
            continue
        left = tx(table_grid.column_boundaries[column_index])
        right = tx(table_grid.column_boundaries[column_index + 1])
        top = ty(row.bounding_box.y0)
        bottom = ty(row.bounding_box.y1)
        draw.rectangle((left, top, right, bottom), outline=(230, 160, 20), width=3)
        draw.text((left + 3, top + 3), "UNRESOLVED", fill=(180, 120, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
