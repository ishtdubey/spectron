import React, { useState } from "react";

export const STAGES = [
  { key: "classify", label: "Classifying query" },
  { key: "plan", label: "Planning perspectives", complexOnly: true },
  { key: "collect", label: "Collecting evidence" },
  { key: "critique", label: "Auditing evidence" },
  { key: "synthesize", label: "Writing report" },
  { key: "epistemic", label: "Updating knowledge state" },
];

/* ------------------------------------------------------------------ */
/* Progress                                                            */
/* ------------------------------------------------------------------ */

/** Live pipeline progress. Shows only real stages the server has reported. */
export function Progress({ stage, queryType, done }) {
  const visible = STAGES.filter((s) => !(s.complexOnly && queryType === "SIMPLE"));
  const activeIdx = visible.findIndex((s) => s.key === stage);
  return (
    <ol className="progress" aria-live="polite">
      {visible.map((s, i) => {
        const state = done || i < activeIdx ? "done" : i === activeIdx ? "active" : "todo";
        return (
          <li key={s.key} className={state}>
            <span className="dot" />
            {s.label}
          </li>
        );
      })}
    </ol>
  );
}

/* ------------------------------------------------------------------ */
/* Safe markdown renderer                                              */
/*                                                                     */
/* Builds React elements only. Nothing is ever inserted as HTML, so    */
/* model output such as "<img onerror=...>" is shown as plain text.    */
/* Supports: # headings, paragraphs, bullet / numbered lists (nested), */
/* tables, > quotes, ``` code ```, --- rules, **bold**, *italic*,      */
/* `code`, and [doc_id] citation chips.                                */
/* ------------------------------------------------------------------ */

const RE_FENCE = /^\s*```/;
const RE_HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const RE_HR = /^\s*([-*_])(\s*\1){2,}\s*$/;
const RE_LIST = /^(\s*)([-*•+]|\d+[.)])\s+(.*)$/;
const RE_QUOTE = /^\s*>\s?(.*)$/;
const RE_TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

function startsBlock(line, next) {
  return (
    RE_FENCE.test(line) ||
    RE_HEADING.test(line) ||
    RE_HR.test(line) ||
    RE_LIST.test(line) ||
    RE_QUOTE.test(line) ||
    (line.includes("|") && next !== undefined && RE_TABLE_SEP.test(next))
  );
}

/** Turn a flat run of list lines into a nested tree by indentation. */
function buildList(items, i, indent) {
  const ordered = /^\d/.test(items[i].marker);
  const node = { ordered, items: [] };
  while (i < items.length && items[i].indent >= indent) {
    if (items[i].indent > indent) {
      // deeper item: attach as a child list of the previous entry
      const [child, next] = buildList(items, i, items[i].indent);
      const last = node.items[node.items.length - 1];
      if (last) last.children.push(child);
      else node.items.push({ text: "", children: [child] });
      i = next;
    } else {
      node.items.push({ text: items[i].text, children: [] });
      i += 1;
    }
  }
  return [node, i];
}

function parseBlocks(src) {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    // fenced code
    if (RE_FENCE.test(line)) {
      const code = [];
      i += 1;
      while (i < lines.length && !RE_FENCE.test(lines[i])) code.push(lines[i++]);
      i += 1; // closing fence (or end of text)
      blocks.push({ type: "code", text: code.join("\n") });
      continue;
    }

    // heading
    let m = line.match(RE_HEADING);
    if (m) {
      blocks.push({ type: "heading", level: m[1].length, text: m[2] });
      i += 1;
      continue;
    }

    // horizontal rule
    if (RE_HR.test(line)) {
      blocks.push({ type: "hr" });
      i += 1;
      continue;
    }

    // table
    if (line.includes("|") && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line);
      const aligns = splitRow(lines[i + 1]).map((c) =>
        c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "left"
      );
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      blocks.push({ type: "table", head, aligns, rows });
      continue;
    }

    // blockquote
    if (RE_QUOTE.test(line)) {
      const q = [];
      while (i < lines.length && RE_QUOTE.test(lines[i])) q.push(lines[i++].match(RE_QUOTE)[1]);
      blocks.push({ type: "quote", text: q.join("\n") });
      continue;
    }

    // list
    if (RE_LIST.test(line)) {
      const items = [];
      while (i < lines.length) {
        const lm = lines[i].match(RE_LIST);
        if (lm) {
          items.push({ indent: lm[1].replace(/\t/g, "    ").length, marker: lm[2], text: lm[3] });
          i += 1;
        } else if (lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && items.length) {
          items[items.length - 1].text += " " + lines[i].trim(); // wrapped continuation
          i += 1;
        } else break;
      }
      const base = Math.min(...items.map((x) => x.indent));
      const [tree] = buildList(items, 0, base);
      blocks.push({ type: "list", tree });
      continue;
    }

    // paragraph: consecutive plain lines
    const para = [line.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) {
      para.push(lines[i].trim());
      i += 1;
    }
    blocks.push({ type: "para", text: para.join(" ") });
  }
  return blocks;
}

/* Inline: `code`, **bold**, __bold__, *italic*, _italic_, [doc_id] citations */
const RE_INLINE =
  /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(\b__[^_\n]+?__\b)|(\*[^*\s][^*\n]*?\*)|(\b_[^_\s][^_\n]*?_\b)|(\[[\w.-]+(?:\s*[,;]\s*[\w.-]+)*\])/g;

function Cite({ id, evidenceById }) {
  const e = evidenceById[id];
  if (!e && !/^doc_/i.test(id)) return <>[{id}]</>; // ordinary bracketed text, not a citation
  return (
    <span
      className={e ? "cite" : "cite unknown"}
      title={e ? `${e.source}: ${e.content}` : "Citation not found in retrieved evidence"}
    >
      {id}
    </span>
  );
}

function Inline({ text, evidenceById }) {
  const out = [];
  let last = 0;
  let k = 0;
  RE_INLINE.lastIndex = 0;
  for (let m = RE_INLINE.exec(text); m; m = RE_INLINE.exec(text)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (m[1]) out.push(<code key={k++}>{tok.slice(1, -1)}</code>);
    else if (m[2] || m[3])
      out.push(
        <strong key={k++}>
          <Inline text={tok.slice(2, -2)} evidenceById={evidenceById} />
        </strong>
      );
    else if (m[4] || m[5])
      out.push(
        <em key={k++}>
          <Inline text={tok.slice(1, -1)} evidenceById={evidenceById} />
        </em>
      );
    else if (m[6]) {
      tok
        .slice(1, -1)
        .split(/\s*[,;]\s*/)
        .forEach((id, j) => out.push(<Cite key={`${k++}-${j}`} id={id} evidenceById={evidenceById} />));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out}</>;
}

function ListTree({ node, evidenceById }) {
  const Tag = node.ordered ? "ol" : "ul";
  return (
    <Tag>
      {node.items.map((it, idx) => (
        <li key={idx}>
          <Inline text={it.text} evidenceById={evidenceById} />
          {it.children.map((c, j) => (
            <ListTree key={j} node={c} evidenceById={evidenceById} />
          ))}
        </li>
      ))}
    </Tag>
  );
}

export function Markdown({ text, evidenceById = {} }) {
  const blocks = parseBlocks(text || "");
  const inl = (t) => <Inline text={t} evidenceById={evidenceById} />;
  return (
    <div className="md">
      {blocks.map((b, i) => {
        switch (b.type) {
          case "heading": {
            const Tag = b.level <= 2 ? "h3" : b.level === 3 ? "h4" : "h5";
            return <Tag key={i}>{inl(b.text)}</Tag>;
          }
          case "hr":
            return <hr key={i} />;
          case "code":
            return (
              <pre key={i}>
                <code>{b.text}</code>
              </pre>
            );
          case "quote":
            return <blockquote key={i}>{inl(b.text)}</blockquote>;
          case "list":
            return <ListTree key={i} node={b.tree} evidenceById={evidenceById} />;
          case "table":
            return (
              <div className="table-wrap" key={i}>
                <table>
                  <thead>
                    <tr>
                      {b.head.map((c, j) => (
                        <th key={j} style={{ textAlign: b.aligns[j] }}>
                          {inl(c)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {b.rows.map((r, ri) => (
                      <tr key={ri}>
                        {b.head.map((_, j) => (
                          <td key={j} style={{ textAlign: b.aligns[j] }}>
                            {inl(r[j] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          default:
            return <p key={i}>{inl(b.text)}</p>;
        }
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Answer card                                                         */
/* ------------------------------------------------------------------ */

function Badge({ children, tone }) {
  return <span className={`badge ${tone || ""}`}>{children}</span>;
}

function List({ items, empty }) {
  if (!items || items.length === 0) return <p className="muted">{empty}</p>;
  return (
    <ul>
      {items.map((x, i) => (
        <li key={i}>{x}</li>
      ))}
    </ul>
  );
}

/** Epistemic state, collapsed by default. */
export function EpistemicPanel({ state }) {
  if (!state) return null;
  const cols = [
    ["Known facts", state.known_facts],
    ["Uncertainties", state.uncertainties],
    ["Disputed claims", state.disputed_claims],
    ["Unresolved questions", state.unresolved_questions],
  ];
  return (
    <details className="panel">
      <summary>Epistemic state</summary>
      <div className="grid">
        {cols.map(([title, items]) => (
          <section key={title}>
            <h4>{title}</h4>
            <List items={items} empty="None reported." />
          </section>
        ))}
      </div>
    </details>
  );
}

/** Evidence list; validated items marked, others flagged. Collapsed by default. */
function EvidencePanel({ evidence, validatedIds }) {
  if (!evidence || evidence.length === 0) return null;
  const valid = new Set(validatedIds || []);
  return (
    <details className="panel">
      <summary>
        Evidence <span className="count">{valid.size} of {evidence.length} validated</span>
      </summary>
      <ul className="evidence">
        {evidence.map((e) => (
          <li key={e.id}>
            <div className="ev-head">
              <code>{e.id}</code>
              {e.perspective && <Badge>{e.perspective}</Badge>}
              <Badge tone={valid.has(e.id) ? "ok" : "warn"}>
                {valid.has(e.id) ? "validated" : "not validated"}
              </Badge>
            </div>
            <div className="muted">{e.source}</div>
            <div>{e.content}</div>
          </li>
        ))}
      </ul>
    </details>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (e.g. insecure context): do nothing */
    }
  }
  return (
    <button type="button" className="ghost" onClick={copy}>
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

/** Final answer: report, concerns, limitations, evidence, epistemic state. */
export function AnswerCard({ result }) {
  const evidenceById = Object.fromEntries((result.evidence || []).map((e) => [e.id, e]));
  const isMulti = result.query_type === "COMPLEX";
  const hasConcerns = result.conflicts?.length > 0 || result.weak_evidence?.length > 0;

  return (
    <article className="answer">
      <div className="answer-head">
        <div className="meta">
          <Badge tone={isMulti ? "accent" : ""}>{isMulti ? "Multi-perspective" : "Direct answer"}</Badge>
          {(result.perspectives || []).map((p) => (
            <Badge key={p}>{p}</Badge>
          ))}
        </div>
        <CopyButton text={result.answer || ""} />
      </div>

      <section className="report">
        {result.answer?.trim() ? (
          <Markdown text={result.answer} evidenceById={evidenceById} />
        ) : (
          <p className="muted">The model returned no text.</p>
        )}
      </section>

      {hasConcerns && (
        <section className="block concerns">
          <h4>Evidence concerns</h4>
          {result.conflicts?.length > 0 && (
            <>
              <h5>Conflicts</h5>
              <List items={result.conflicts} empty="" />
            </>
          )}
          {result.weak_evidence?.length > 0 && (
            <>
              <h5>Weak evidence</h5>
              <List items={result.weak_evidence} empty="" />
            </>
          )}
        </section>
      )}

      <section className="block limits">
        <h4>Limitations</h4>
        <List items={result.limitations} empty="The pipeline reported no limitations." />
      </section>

      <div className="details">
        <EvidencePanel evidence={result.evidence} validatedIds={result.validated_evidence_ids} />
        <EpistemicPanel state={result.epistemic} />
      </div>
    </article>
  );
}
