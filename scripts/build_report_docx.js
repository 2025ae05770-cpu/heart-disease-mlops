// Build a draft .docx (reports/Heart_Disease_MLOps_Report_draft.docx) from reports/REPORT.md (docx-js).
// The submitted report (reports/Heart_Disease_MLOPS_Report.docx) is finalised by hand in Word,
// so the default output uses a different name and never overwrites it (Windows paths are case-insensitive).
// Usage:  node scripts/build_report_docx.js [output.docx]
// Supports the Markdown subset used in REPORT.md: headings, paragraphs, bullet / numbered
// lists, pipe tables, fenced code, images and inline **bold**, `code`, _italic_, <links>.
// Image lines with no blank line between them are laid out side by side (2 per row).
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

let docx;
try {
  docx = require("docx");
} catch {
  docx = require(path.join(execSync("npm root -g").toString().trim(), "docx"));
}
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Header, Footer,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
  ExternalHyperlink, TabStopType,
} = docx;

const ROOT = path.resolve(__dirname, "..");
const MD_PATH = path.join(ROOT, "reports", "REPORT.md");
const OUT_PATH = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, "reports", "Heart_Disease_MLOps_Report_draft.docx");

const PAGE = { width: 11906, height: 16838, margin: 1020 }; // A4, 1.8 cm margins
const CONTENT_W = PAGE.width - 2 * PAGE.margin; // 9638 DXA
const CONTENT_IN = CONTENT_W / 1440;
const FONT = "Arial";
const ACCENT = "1F4E79";

// ---------- inline formatting ----------
function inlineRuns(text, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|_<[^>]+>_|<https?:[^>]+>|(?<!\w)_[^_\s][^_]*_(?!\w))/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    } else if (tok.startsWith("`")) {
      runs.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", color: "8B1A1A", ...base,
        size: base.size ? base.size - 2 : 19 }));
    } else if (tok.startsWith("<http")) {
      const url = tok.slice(1, -1);
      runs.push(new ExternalHyperlink({ link: url, children: [new TextRun({ text: url, style: "Hyperlink", ...base })] }));
    } else {
      runs.push(new TextRun({ text: tok.slice(1, -1).replace(/^<|>$/g, ""), italics: true, ...base }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}

// ---------- images ----------
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), data: buf };
}

function targetWidthIn(file) {
  const name = path.basename(file);
  if (/class_balance|confusion_matrix|feature_importance|comparison_roc|missing_values/.test(name)) return 3.6;
  if (/^\d\d_.*\.png$/.test(name) && file.includes("screenshots")) return 3.9;
  if (/architecture/.test(name)) return 5.4;
  return 5.0;
}

function imageBlock(alt, relPath, opts = {}) {
  const file = path.resolve(path.dirname(MD_PATH), relPath);
  const caption = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: opts.captionAfter ?? 160 },
    children: [new TextRun({ text: alt, italics: true, size: 17, color: "555555" })],
  });
  if (!fs.existsSync(file)) {
    const rel = path.relative(ROOT, file).replace(/\\/g, "/");
    const border = { style: BorderStyle.DASHED, size: 6, color: "C55A11", space: 6 };
    return [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 60 },
        border: { top: border, bottom: border },
        shading: { fill: "FFF4E5", type: ShadingType.CLEAR },
        children: [
          new TextRun({ text: "Screenshot to add: ", bold: true, color: "C55A11" }),
          new TextRun({ text: rel, font: "Consolas", color: "C55A11" }),
          new TextRun({ text: "  (save the PNG there and re-run node scripts/build_report_docx.js)", color: "7F7F7F", size: 18 }),
        ],
      }),
      caption,
    ];
  }
  const { w, h, data } = pngSize(file);
  let widthIn = Math.min(opts.maxW ?? targetWidthIn(file), CONTENT_IN);
  let heightIn = (widthIn * h) / w;
  const maxH = opts.maxH ?? 3.8;
  if (heightIn > maxH) { widthIn *= maxH / heightIn; heightIn = maxH; }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: opts.before ?? 80 },
      children: [new ImageRun({
        type: "png", data,
        transformation: { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) },
        altText: { title: alt, description: alt, name: path.basename(file) },
      })],
    }),
    caption,
  ];
}

// Figures side by side in a borderless table (one row per pair).
function imageRow(images) {
  const n = images.length;
  const colW = Math.floor(CONTENT_W / n);
  const widths = Array(n).fill(colW);
  widths[n - 1] += CONTENT_W - colW * n;
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const borders = { top: none, bottom: none, left: none, right: none };
  return [
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: widths,
      borders: { ...borders, insideHorizontal: none, insideVertical: none },
      rows: [new TableRow({
        cantSplit: true,
        children: images.map(([alt, rel], i) => new TableCell({
          borders,
          width: { size: widths[i], type: WidthType.DXA },
          margins: { top: 0, bottom: 0, left: 60, right: 60 },
          children: imageBlock(alt, rel, { maxW: widths[i] / 1440 - 0.15, maxH: 2.35, before: 40, captionAfter: 60 }),
        })),
      })],
    }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}

// ---------- tables ----------
function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function tableBlock(lines) {
  const header = splitRow(lines[0]);
  const rows = lines.slice(2).map(splitRow);
  const n = header.length;
  const fontSize = n >= 9 ? 14 : n >= 5 ? 16 : 18;
  const plain = (s) => s.replace(/\*\*|`|_/g, "");
  const lens = header.map((h, i) => Math.max(plain(h).length, ...rows.map((r) => plain(r[i] || "").length)));
  const weights = lens.map((l) => Math.min(Math.max(l, 6), 60));
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((wt) => Math.floor((wt / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);

  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, isHeader, stripe) => new TableCell({
    borders,
    width: { size: widths[i], type: WidthType.DXA },
    shading: isHeader ? { fill: ACCENT, type: ShadingType.CLEAR } : stripe ? { fill: "F2F6FA", type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({
      children: inlineRuns(text, isHeader ? { bold: true, color: "FFFFFF", size: fontSize } : { size: fontSize }),
    })],
  });
  return [
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false)) }),
        ...rows.map((r, ri) => new TableRow({ children: header.map((_, i) => cell(r[i] || "", i, false, ri % 2 === 1)) })),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---------- code ----------
function codeBlock(lines) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "D0D7DE", space: 4 };
  return lines.map((line, i) => new Paragraph({
    shading: { fill: "F6F8FA", type: ShadingType.CLEAR },
    border: {
      ...(i === 0 ? { top: border } : {}),
      ...(i === lines.length - 1 ? { bottom: border } : {}),
    },
    spacing: { before: 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
    children: [new TextRun({ text: line || " ", font: "Consolas", size: 16 })],
  }));
}

// ---------- document ----------
function build() {
  const lines = fs.readFileSync(MD_PATH, "utf8").split(/\r?\n/);
  const children = [];
  const numberedRefs = [];
  let currentNumbered = null;
  let title = "Report";
  let i = 0;
  let h2Count = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { currentNumbered = null; i++; continue; }

    if (line.startsWith("# ")) {
      title = line.slice(2).trim();
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: title, bold: true, size: 36, color: ACCENT })],
      }));
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 8 } },
        children: [new TextRun({ text: "UCI Heart Disease  |  MLflow  |  FastAPI  |  Docker  |  Kubernetes  |  Prometheus/Grafana", size: 20, color: "595959" })],
      }));
      i++; continue;
    }
    if (line.startsWith("## ")) {
      h2Count++;
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun(line.slice(3).trim())],
      }));
      i++; continue;
    }
    if (line.startsWith("### ")) {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(line.slice(4).trim())] }));
      i++; continue;
    }
    if (line.startsWith("```")) {
      const block = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) block.push(lines[i++]);
      i++;
      children.push(...codeBlock(block));
      continue;
    }
    if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      children.push(...tableBlock(block));
      continue;
    }
    const IMG_RE = /^!\[(.*)\]\((.*)\)\s*$/;
    if (IMG_RE.test(line)) {
      const group = [];
      while (i < lines.length && IMG_RE.test(lines[i])) {
        const m = lines[i].match(IMG_RE);
        group.push([m[1], m[2]]);
        i++;
      }
      if (group.length === 1) children.push(...imageBlock(...group[0]));
      for (let g = 0; group.length > 1 && g < group.length; g += 2) {
        const pair = group.slice(g, g + 2);
        children.push(...(pair.length === 2 ? imageRow(pair) : imageBlock(...pair[0])));
      }
      continue;
    }

    const bullet = line.match(/^(\s*)- (.*)$/);
    if (bullet) {
      const level = Math.min(Math.floor(bullet[1].length / 2), 1);
      children.push(new Paragraph({
        numbering: { reference: "bullets", level },
        spacing: { after: 60 },
        children: inlineRuns(bullet[2]),
      }));
      i++; continue;
    }
    const numbered = line.match(/^\d+\. (.*)$/);
    if (numbered) {
      if (!currentNumbered) { currentNumbered = `numbers-${numberedRefs.length + 1}`; numberedRefs.push(currentNumbered); }
      children.push(new Paragraph({
        numbering: { reference: currentNumbered, level: 0 },
        spacing: { after: 60 },
        children: inlineRuns(numbered[1]),
      }));
      i++; continue;
    }
    const isMeta = /^\*\*[^*]+:\*\*/.test(line) && h2Count === 0;
    children.push(new Paragraph({
      alignment: isMeta ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
      spacing: { after: isMeta ? 40 : 100 },
      children: inlineRuns(line),
    }));
    i++;
  }

  const numberingConfig = [
    {
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 270 } } } },
      ],
    },
    ...numberedRefs.map((ref) => ({
      reference: ref,
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }],
    })),
  ];

  const doc = new Document({
    creator: "heart-disease-mlops",
    title,
    description: "MLOps Assignment 1 report",
    styles: {
      default: { document: { run: { font: FONT, size: 20 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: FONT, color: ACCENT },
          paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 0, keepNext: true,
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "9DC3E6", space: 4 } } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: FONT, color: "2E75B6" },
          paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1, keepNext: true } },
      ],
    },
    numbering: { config: numberingConfig },
    sections: [{
      properties: {
        page: {
          size: { width: PAGE.width, height: PAGE.height },
          margin: { top: PAGE.margin, right: PAGE.margin, bottom: PAGE.margin, left: PAGE.margin },
        },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "AIMLCZG523 MLOps - Assignment 1 - Heart Disease Risk Prediction", size: 16, color: "7F7F7F" })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
          children: [
            new TextRun({ text: "heart-disease-mlops", size: 16, color: "7F7F7F" }),
            new TextRun({ children: ["\tPage ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "7F7F7F" }),
          ],
        })] }),
      },
      children,
    }],
  });

  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(OUT_PATH, buf);
    console.log(`Wrote ${OUT_PATH} (${(buf.length / 1024).toFixed(0)} KB)`);
  });
}

build().catch((err) => { console.error(err); process.exit(1); });
