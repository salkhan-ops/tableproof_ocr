"""Synthetic financial-statement-style tables for the TableProof Phase A fitness test.

All entities and amounts are fabricated. Produces text PDFs (reportlab) plus a ground-truth
cell matrix per table (JSON). Ground truth holds the cell text exactly as rendered.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = A4
FONT, BOLD = "Helvetica", "Helvetica-Bold"
SIZE = 9
LEAD = 13.5


@dataclass
class Row:
    label: str
    values: list[str] = field(default_factory=list)
    note: str = ""
    kind: str = "item"  # header | units | section | item | subtotal | total
    indent: int = 0


@dataclass
class Table:
    table_id: str
    title: list[str]
    columns: list[float]  # right edges of value columns (points)
    note_x: float | None  # centre of the note column, None if no note column
    rows: list[Row]
    page_break_after: int | None = None  # row index after which the table continues on page 2
    hanging_brackets: bool = False  # closing ')' hangs past the column's right edge
    footnotes: list[str] = field(default_factory=list)
    stub_header: str = ""


def _fmt_cells(table: Table, row: Row) -> list[str]:
    cells = [row.label]
    if table.note_x is not None:
        cells.append(row.note)
    vals = list(row.values) + [""] * (len(table.columns) - len(row.values))
    return cells + vals


def ground_truth(table: Table) -> dict:
    rows = [_fmt_cells(table, r) for r in table.rows]
    pages: list[list[list[str]]] = []
    if table.page_break_after is None:
        pages = [rows]
    else:
        header = [_fmt_cells(table, r) for r in table.rows if r.kind in {"header", "units"}]
        first = rows[: table.page_break_after + 1]
        second = header + rows[table.page_break_after + 1 :]
        pages = [first, second]
    return {
        "table_id": table.table_id,
        "kinds": [r.kind for r in table.rows],
        "pages": pages,
        "title": table.title,
        "footnotes": table.footnotes,
    }


def _draw_rows(c: canvas.Canvas, table: Table, rows: list[Row], y: float) -> float:
    left = 56.0
    for row in rows:
        font = BOLD if row.kind in {"header", "total", "section", "units"} else FONT
        c.setFont(font, SIZE)
        if row.label:
            c.drawString(left + 12 * row.indent, y, row.label)
        if table.note_x is not None and row.note:
            c.drawCentredString(table.note_x, y, row.note)
        for x_right, value in zip(table.columns, row.values):
            if not value:
                continue
            if table.hanging_brackets and value.endswith(")"):
                c.drawRightString(x_right + c.stringWidth(")", font, SIZE), y, value)
            elif value in {"-", "–"}:
                c.drawRightString(x_right - 4, y, value)
            else:
                c.drawRightString(x_right, y, value)
        if row.kind in {"subtotal", "total"}:
            c.setLineWidth(0.6)
            for x_right in table.columns:
                c.line(x_right - 52, y + SIZE + 1.5, x_right + 2, y + SIZE + 1.5)  # rule above
            if row.kind == "total":
                for x_right in table.columns:
                    c.line(x_right - 52, y - 3, x_right + 2, y - 3)  # double rule below
                    c.line(x_right - 52, y - 5, x_right + 2, y - 5)
        y -= LEAD
    return y


def render(table: Table, out_dir: Path) -> Path:
    path = out_dir / f"{table.table_id}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Synthetic statement")
    c.setAuthor("synthetic")

    def header_block(cont: bool) -> float:
        y = PAGE_H - 60
        for i, line in enumerate(table.title):
            c.setFont(BOLD if i == 0 else FONT, 11 if i == 0 else 9)
            c.drawString(56, y, line + (" (continued)" if cont and i == 1 else ""))
            y -= 15
        return y - 12

    if table.page_break_after is None:
        y = header_block(False)
        y = _draw_rows(c, table, table.rows, y)
        _draw_footnotes(c, table, y)
    else:
        y = header_block(False)
        _draw_rows(c, table, table.rows[: table.page_break_after + 1], y)
        c.setFont(FONT, 8)
        c.drawString(56, 40, "Continued on next page")
        c.showPage()
        y = header_block(True)
        header = [r for r in table.rows if r.kind in {"header", "units"}]
        y = _draw_rows(c, table, header + table.rows[table.page_break_after + 1 :], y)
        _draw_footnotes(c, table, y)
    c.showPage()
    c.save()
    return path


def _draw_footnotes(c: canvas.Canvas, table: Table, y: float) -> None:
    y -= 10
    c.setFont(FONT, 7.5)
    for note in table.footnotes:
        c.drawString(56, y, note)
        y -= 11


H = "header"
U = "units"
S = "section"
T = "total"
ST = "subtotal"


def tables() -> list[Table]:
    two = [430.0, 520.0]
    three = [380.0, 455.0, 530.0]
    sofp = Table(
        "fs01_sofp_two_period",
        ["Synthetic Holdings Limited", "Statement of financial position", "As at 30 June 2025"],
        two,
        330.0,
        [
            Row("", ["2025", "2024"], "Note", H),
            Row("", ["Rs. '000", "Rs. '000"], "", U),
            Row("ASSETS", kind=S),
            Row("Non-current assets", kind=S),
            Row("Property, plant and equipment", ["1,245,300", "1,198,450"], "5", indent=1),
            Row("Intangible assets", ["84,120", "91,005"], "6", indent=1),
            Row("Long-term deposits", ["-", "2,500"], "", indent=1),
            Row("", ["1,329,420", "1,291,955"], "", ST),
            Row("Current assets", kind=S),
            Row("Inventories", ["412,880", "389,214"], "7", indent=1),
            Row("Trade and other receivables", ["298,045", "275,330"], "8", indent=1),
            Row("Cash and cash equivalents", ["76,511", "102,906"], "9", indent=1),
            Row("", ["787,436", "767,450"], "", ST),
            Row("Total assets", ["2,116,856", "2,059,405"], "", T),
            Row("EQUITY AND LIABILITIES", kind=S),
            Row("Share capital", ["500,000", "500,000"], "10", indent=1),
            Row("Retained earnings", ["842,611", "781,240"], "", indent=1),
            Row("Revaluation reserve", ["(12,450)", "(8,900)"], "", indent=1),
            Row("Total equity", ["1,330,161", "1,272,340"], "", ST),
            Row("Long-term financing", ["410,000", "455,000"], "11", indent=1),
            Row("Trade and other payables", ["376,695", "332,065"], "12", indent=1),
            Row("Total equity and liabilities", ["2,116,856", "2,059,405"], "", T),
        ],
    )
    pl = Table(
        "fs02_pl_three_period",
        ["Synthetic Traders (Private) Limited", "Statement of profit or loss", "For the year ended 31 December 2025"],
        three,
        310.0,
        [
            Row("", ["2025", "2024", "2023"], "Note", H),
            Row("", ["$'000", "$'000", "$'000"], "", U),
            Row("Revenue", ["8,402,115", "7,950,330", "7,110,004"], "13"),
            Row("Cost of sales", ["(5,911,420)", "(5,602,118)", "(5,020,775)"], "14"),
            Row("Gross profit", ["2,490,695", "2,348,212", "2,089,229"], "", ST),
            Row("Distribution costs", ["(612,004)", "(588,140)", "(540,110)"], ""),
            Row("Administrative expenses", ["(431,950)", "(410,662)", "(398,001)"], ""),
            Row("Other income", ["15,220", "–", "9,875"], "15"),
            Row("Operating profit", ["1,461,961", "1,349,410", "1,160,993"], "", ST),
            Row("Finance costs", ["(88,412)", "(95,003)", "(101,540)"], "16"),
            Row("Profit before tax", ["1,373,549", "1,254,407", "1,059,453"], "", ST),
            Row("Taxation", ["(398,329)", "(363,778)", "(307,241)"], "17"),
            Row("Profit for the year", ["975,220", "890,629", "752,212"], "", T),
            Row("Earnings per share - basic (in $)", ["9.75", "8.91", "7.52"], "18"),
        ],
    )
    cf = Table(
        "fs03_cash_flow_hanging_brackets",
        ["Synthetic Industries Plc", "Statement of cash flows", "For the year ended 30 June 2025"],
        two,
        330.0,
        [
            Row("", ["2025", "2024"], "Note", H),
            Row("", ["Rs. '000", "Rs. '000"], "", U),
            Row("Cash flows from operating activities", kind=S),
            Row("Profit before tax", ["1,373,549", "1,254,407"], "", indent=1),
            Row("Adjustments for:", kind="item", indent=1),
            Row("Depreciation", ["145,200", "139,880"], "5", indent=2),
            Row("Finance costs", ["88,412", "95,003"], "", indent=2),
            Row("Increase in inventories", ["(23,666)", "(41,207)"], "", indent=2),
            Row("Income tax paid", ["(401,210)", "(352,904)"], "", indent=2),
            Row("Net cash generated from operating activities", ["1,182,285", "1,095,179"], "", ST),
            Row("Cash flows from investing activities", kind=S),
            Row("Purchase of property, plant and equipment", ["(192,050)", "(176,300)"], "", indent=1),
            Row("Proceeds from disposal", ["4,100", "-"], "", indent=1),
            Row("Net cash used in investing activities", ["(187,950)", "(176,300)"], "", ST),
            Row("Net (decrease) / increase in cash", ["(26,395)", "18,207"], "", ST),
            Row("Cash at beginning of the year", ["102,906", "84,699"], "", indent=1),
            Row("Cash at end of the year", ["76,511", "102,906"], "9", T),
        ],
        hanging_brackets=True,
    )
    soce = Table(
        "fs04_changes_in_equity",
        ["Synthetic Holdings Limited", "Statement of changes in equity", "For the year ended 30 June 2025"],
        [250.0, 345.0, 440.0, 530.0],
        None,
        [
            Row("", ["Share capital", "Retained earnings", "Revaluation reserve", "Total"], "", H),
            Row("", ["Rs. '000", "Rs. '000", "Rs. '000", "Rs. '000"], "", U),
            Row("Balance at 1 July 2023", ["500,000", "702,118", "(6,200)", "1,195,918"]),
            Row("Profit for the year", ["-", "129,122", "-", "129,122"]),
            Row("Other comprehensive loss", ["-", "-", "(2,700)", "(2,700)"]),
            Row("Dividend paid", ["-", "(50,000)", "-", "(50,000)"]),
            Row("Balance at 30 June 2024", ["500,000", "781,240", "(8,900)", "1,272,340"], kind=T),
            Row("Balance at 1 July 2024", ["500,000", "781,240", "(8,900)", "1,272,340"]),
            Row("Profit for the year", ["-", "111,371", "-", "111,371"]),
            Row("Other comprehensive loss", ["-", "-", "(3,550)", "(3,550)"]),
            Row("Dividend paid", ["-", "(50,000)", "-", "(50,000)"]),
            Row("Balance at 30 June 2025", ["500,000", "842,611", "(12,450)", "1,330,161"], kind=T),
        ],
    )
    two_page = Table(
        "fs05_sofp_two_pages",
        ["Synthetic Group Limited", "Consolidated statement of financial position", "As at 31 March 2025"],
        two,
        330.0,
        [
            Row("", ["2025", "2024"], "Note", H),
            Row("", ["Rs. '000", "Rs. '000"], "", U),
            Row("Non-current assets", kind=S),
            Row("Property, plant and equipment", ["3,410,200", "3,155,870"], "4", indent=1),
            Row("Right-of-use assets", ["211,450", "240,115"], "5", indent=1),
            Row("Investment property", ["98,000", "98,000"], "6", indent=1),
            Row("Deferred tax asset", ["-", "12,300"], "7", indent=1),
            Row("", ["3,719,650", "3,506,285"], "", ST),
            Row("Current assets", kind=S),
            Row("Inventories", ["1,020,400", "988,210"], "8", indent=1),
            Row("Trade receivables", ["640,118", "611,909"], "9", indent=1),
            Row("Cash and bank balances", ["155,032", "130,446"], "10", indent=1),
            Row("", ["1,815,550", "1,730,565"], "", ST),
            Row("Total assets", ["5,535,200", "5,236,850"], "", T),
            Row("Equity", kind=S),
            Row("Share capital", ["1,000,000", "1,000,000"], "11", indent=1),
            Row("Unappropriated profit", ["2,104,560", "1,890,300"], "", indent=1),
            Row("", ["3,104,560", "2,890,300"], "", ST),
            Row("Non-current liabilities", kind=S),
            Row("Long-term loans", ["1,200,000", "1,350,000"], "12", indent=1),
            Row("Lease liabilities", ["180,640", "205,550"], "13", indent=1),
            Row("Current liabilities", kind=S),
            Row("Trade and other payables", ["1,050,000", "791,000"], "14", indent=1),
            Row("Total equity and liabilities", ["5,535,200", "5,236,850"], "", T),
        ],
        page_break_after=13,
    )
    notes = Table(
        "fs06_note_ppe_currency",
        ["Synthetic Holdings Limited", "Notes to the financial statements", "5  Property, plant and equipment"],
        [250.0, 345.0, 440.0, 530.0],
        None,
        [
            Row("", ["Land", "Buildings", "Plant and machinery", "Total"], "", H),
            Row("", ["USD", "USD", "USD", "USD"], "", U),
            Row("Cost", kind=S),
            Row("At 1 July 2024", ["$ 120,000", "$ 455,300", "$ 1,302,800", "$ 1,878,100"], indent=1),
            Row("Additions", ["-", "12,450", "179,600", "192,050"], indent=1),
            Row("Disposals", ["-", "-", "(8,300)", "(8,300)"], indent=1),
            Row("At 30 June 2025", ["$ 120,000", "$ 467,750", "$ 1,474,100", "$ 2,061,850"], kind=ST),
            Row("Accumulated depreciation", kind=S),
            Row("At 1 July 2024", ["-", "(152,400)", "(527,250)", "(679,650)"], indent=1),
            Row("Charge for the year *", ["-", "(22,700)", "(122,500)", "(145,200)"], indent=1),
            Row("On disposals", ["-", "-", "4,200", "4,200"], indent=1),
            Row("At 30 June 2025", ["-", "(175,100)", "(645,550)", "(820,650)"], kind=ST),
            Row("Net book value at 30 June 2025", ["$ 120,000", "$ 292,650", "$ 828,550", "$ 1,241,200"], kind=T),
        ],
        footnotes=["* Includes Rs. 3,100 thousand charged to cost of sales.", "Source: synthetic data, no real entity."],
    )
    return [sofp, pl, cf, soce, two_page, notes]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for table in tables():
        render(table, out)
        (out / f"{table.table_id}.gt.json").write_text(json.dumps(ground_truth(table), indent=2))
    print(f"wrote {len(tables())} tables to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
