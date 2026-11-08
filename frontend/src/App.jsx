import React, { useEffect, useRef, useState } from "react";
import { streamQuery } from "./api.js";
import { AnswerCard, Progress } from "./components.jsx";

const EXAMPLES = [
  "What is the capital of Australia?",
  "What are the trade-offs of moving a core financial system from microservices back to a modular monolith?",
];

let nextId = 1;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const patch = (id, fn) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...fn(m) } : m)));

  async function send(text) {
    const query = text.trim();
    if (!query || busy) return;

    const aid = nextId++;
    setMessages((ms) => [
      ...ms,
      { id: nextId++, role: "user", text: query },
      { id: aid, role: "assistant", status: "running", stage: "classify", queryType: null },
    ]);
    setInput("");
    setBusy(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let finished = false;

    try {
      await streamQuery(
        query,
        (e) => {
          if (e.type === "stage") patch(aid, () => ({ stage: e.stage }));
          else if (e.type === "classified") patch(aid, () => ({ queryType: e.query_type }));
          else if (e.type === "result") {
            finished = true;
            patch(aid, () => ({ status: "done", result: e }));
          } else if (e.type === "error") {
            finished = true;
            patch(aid, () => ({ status: "error", error: e.message }));
          }
        },
        ctrl.signal
      );
      if (!finished) {
        patch(aid, () => ({ status: "error", error: "The connection closed before an answer arrived." }));
      }
    } catch (err) {
      if (err.name === "AbortError") {
        patch(aid, () => ({ status: "error", error: "Cancelled." }));
      } else {
        patch(aid, () => ({
          status: "error",
          error: `Could not reach the backend (${err.message}). Is it running on port 8000?`,
        }));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    send(input);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Research Agent</h1>
        <p className="muted">Multi-perspective answers with evidence checks and stated limitations.</p>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask a question. Simple facts get a direct answer; open-ended ones are split into perspectives.</p>
            {EXAMPLES.map((ex) => (
              <button key={ex} className="example" onClick={() => send(ex)} disabled={busy}>
                {ex}
              </button>
            ))}
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg user">
              {m.text}
            </div>
          ) : (
            <div key={m.id} className="msg bot">
              {m.status === "running" && <Progress stage={m.stage} queryType={m.queryType} />}
              {m.status === "error" && (
                <div className="error" role="alert">
                  {m.error}
                </div>
              )}
              {m.status === "done" && <AnswerCard result={m.result} />}
            </div>
          )
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a research question…"
          rows={2}
          maxLength={2000}
          disabled={busy}
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Ask
          </button>
        )}
      </form>
    </div>
  );
}
