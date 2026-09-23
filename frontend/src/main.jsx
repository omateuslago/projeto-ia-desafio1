import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleAlert,
  FileText,
  Home,
  Loader2,
  Mic,
  Pause,
  Play,
  RefreshCw,
} from "lucide-react";
import "./styles.css";

const POLL_INTERVAL = 3000;

const pages = [
  { id: "inicio", label: "Início", icon: Home },
  { id: "ao-vivo", label: "Reunião ao Vivo", icon: Mic },
  { id: "relatorio", label: "Relatório", icon: FileText },
];

function api(path, options = {}) {
  return fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  }).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail || data.mensagem || "Falha na requisição.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  });
}

function formatSeconds(value) {
  const seconds = Number(value || 0);
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${(seconds / 60).toFixed(2)} min`;
}

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${Math.round(numeric * 100)}%`;
}

function formatDate(value) {
  if (!value) return "Sem registros";
  return new Intl.DateTimeFormat("pt-BR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function useHashPage() {
  const readPage = () => window.location.hash.replace("#", "") || "inicio";
  const [page, setPage] = useState(readPage);

  useEffect(() => {
    const onHashChange = () => setPage(readPage());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return pages.some((item) => item.id === page) ? page : "inicio";
}

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ title, text }) {
  return (
    <div className="empty-state">
      <CircleAlert size={22} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <p>{text}</p>
      </div>
    </div>
  );
}

function StatusPill({ ok, label }) {
  return (
    <span className={`pill ${ok ? "ok" : "warn"}`}>
      {ok ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
      {label}
    </span>
  );
}

function MeetingSummary({ meeting }) {
  if (!meeting) {
    return (
      <EmptyState
        title="Nenhuma reunião encontrada"
        text="A interface vai mostrar dados assim que os scripts criarem uma pasta em reunioes/ com registros.csv."
      />
    );
  }

  const summary = meeting.resumo;

  return (
    <div className="summary-grid">
      <Stat label="Reunião" value={meeting.nome} />
      <Stat label="Trechos" value={summary.trechos} />
      <Stat label="Duração" value={formatSeconds(summary.duracao_segundos)} />
      <Stat label="Participantes" value={summary.participantes.length} />
      <Stat label="Relatório" value={meeting.tem_relatorio ? "Gerado" : "Pendente"} tone={meeting.tem_relatorio ? "good" : "quiet"} />
      <Stat label="Atualizado" value={formatDate(meeting.modificado_em)} />
    </div>
  );
}

function RecordsTable({ rows }) {
  if (!rows?.length) {
    return (
      <EmptyState
        title="Sem linhas no CSV"
        text="Quando a análise gravar blocos de áudio, os registros aparecerão aqui."
      />
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Início</th>
            <th>Fim</th>
            <th>Pessoa</th>
            <th>Confiança</th>
            <th>Emoção</th>
            <th>Confiança</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.arquivo || index}-${index}`}>
              <td>{formatSeconds(row.inicio)}</td>
              <td>{formatSeconds(row.fim)}</td>
              <td>{row.pessoa || "-"}</td>
              <td>{formatPercent(row.conf_pessoa)}</td>
              <td>{row.emocao || "-"}</td>
              <td>{formatPercent(row.conf_emocao)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HomePage({ status, onRefresh }) {
  const latest = status?.reuniao_mais_recente;
  const modelsReady = Boolean(status?.modelos?.pessoas && status?.modelos?.emocoes);

  return (
    <main className="page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Painel local</p>
          <h1>Sala de Reunião Inteligente</h1>
        </div>
        <button className="icon-button" onClick={onRefresh} title="Atualizar agora">
          <RefreshCw size={18} />
        </button>
      </section>

      <section className="status-row">
        <StatusPill ok={status?.scripts?.analisar_reuniao} label="Script de análise" />
        <StatusPill ok={status?.scripts?.gerar_relatorio} label="Script de relatório" />
        <StatusPill ok={modelsReady} label="Modelos treinados" />
        <StatusPill ok={status?.processo?.rodando} label={status?.processo?.rodando ? "Análise rodando" : "Análise parada"} />
      </section>

      <section className="panel">
        <div className="section-title">
          <Activity size={19} />
          <h2>Reunião mais recente</h2>
        </div>
        <MeetingSummary meeting={latest} />
      </section>

      <section className="panel">
        <div className="section-title">
          <BarChart3 size={19} />
          <h2>Distribuição de emoções</h2>
        </div>
        {latest?.resumo?.emocoes && Object.keys(latest.resumo.emocoes).length ? (
          <div className="emotion-list">
            {Object.entries(latest.resumo.emocoes).map(([emotion, count]) => (
              <div className="emotion-item" key={emotion}>
                <span>{emotion}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="Sem emoções registradas"
            text="Nenhum dado real foi encontrado no CSV da reunião mais recente."
          />
        )}
      </section>
    </main>
  );
}

function LivePage({ status, latest, onRefresh }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const start = async () => {
    setBusy(true);
    setError("");
    try {
      await api("/api/reunioes/analisar/iniciar", { method: "POST" });
      await onRefresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setError("");
    try {
      await api("/api/reunioes/analisar/parar", { method: "POST" });
      await onRefresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const running = Boolean(status?.processo?.rodando);
  const logs = status?.processo?.logs || [];

  return (
    <main className="page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Atualização a cada 3 segundos</p>
          <h1>Reunião ao Vivo</h1>
        </div>
        <div className="actions">
          <button className="primary" onClick={start} disabled={busy || running}>
            {busy && !running ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Iniciar
          </button>
          <button className="secondary" onClick={stop} disabled={busy || !running}>
            <Pause size={18} />
            Parar
          </button>
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}

      <section className="panel">
        <div className="section-title">
          <Mic size={19} />
          <h2>Status da captura</h2>
        </div>
        <MeetingSummary meeting={latest} />
      </section>

      <section className="panel">
        <div className="section-title">
          <Activity size={19} />
          <h2>Últimos registros do CSV</h2>
        </div>
        <RecordsTable rows={latest?.registros || []} />
      </section>

      <section className="panel">
        <div className="section-title">
          <FileText size={19} />
          <h2>Saída do script</h2>
        </div>
        {logs.length ? (
          <pre className="logs">{logs.slice(-40).join("\n")}</pre>
        ) : (
          <EmptyState
            title="Sem saída capturada"
            text="Ao iniciar a análise, as mensagens reais do script aparecerão nesta área."
          />
        )}
      </section>
    </main>
  );
}

function ReportPage({ meetings, selectedName, setSelectedName, selectedMeeting, onRefresh }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const generate = async () => {
    if (!selectedName) return;
    setBusy(true);
    setError("");
    try {
      await api(`/api/reunioes/${selectedName}/relatorio`, { method: "POST" });
      await onRefresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Consolidado gerado pelos scripts</p>
          <h1>Relatório</h1>
        </div>
        <div className="actions">
          <select
            value={selectedName || ""}
            onChange={(event) => setSelectedName(event.target.value)}
          >
            {meetings.map((meeting) => (
              <option key={meeting.nome} value={meeting.nome}>
                {meeting.nome}
              </option>
            ))}
          </select>
          <button className="primary" onClick={generate} disabled={!selectedName || busy}>
            {busy ? <Loader2 className="spin" size={18} /> : <FileText size={18} />}
            Gerar
          </button>
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}

      <section className="panel">
        <div className="section-title">
          <BarChart3 size={19} />
          <h2>Dados selecionados</h2>
        </div>
        <MeetingSummary meeting={selectedMeeting} />
        {selectedMeeting?.tem_csv ? (
          <div className="file-links">
            <a href={selectedMeeting.registros_url} target="_blank" rel="noreferrer">
              registros.csv
            </a>
            {selectedMeeting.relatorio_txt_url ? (
              <a href={selectedMeeting.relatorio_txt_url} target="_blank" rel="noreferrer">
                relatorio.txt
              </a>
            ) : null}
            {selectedMeeting.baixa_confianca_url ? (
              <a href={selectedMeeting.baixa_confianca_url} target="_blank" rel="noreferrer">
                baixa_confianca.csv
              </a>
            ) : null}
          </div>
        ) : null}
      </section>

      {selectedMeeting?.tem_relatorio ? (
        <section className="report-layout">
          <div className="panel report-frame-panel">
            <div className="section-title">
              <FileText size={19} />
              <h2>Relatório HTML</h2>
            </div>
            <iframe title="Relatório da reunião" src={selectedMeeting.relatorio_url} />
          </div>
          <div className="panel">
            <div className="section-title">
              <BarChart3 size={19} />
              <h2>Gráficos gerados</h2>
            </div>
            {selectedMeeting.graficos.length ? (
              <div className="graphs">
                {selectedMeeting.graficos.map((graph) => (
                  <figure key={graph.nome}>
                    <img src={graph.url} alt={graph.nome} />
                    <figcaption>{graph.nome}</figcaption>
                  </figure>
                ))}
              </div>
            ) : (
              <EmptyState
                title="Sem gráficos"
                text="O relatório ainda não gerou imagens para esta reunião."
              />
            )}
          </div>
          <div className="panel report-text-panel">
            <div className="section-title">
              <FileText size={19} />
              <h2>Resumo em texto</h2>
            </div>
            {selectedMeeting.relatorio_texto ? (
              <pre className="report-text">{selectedMeeting.relatorio_texto}</pre>
            ) : (
              <EmptyState
                title="Sem relatório TXT"
                text="O arquivo relatorio.txt ainda não foi encontrado para esta reunião."
              />
            )}
          </div>
        </section>
      ) : (
        <section className="panel">
          <EmptyState
            title="Relatório ainda não gerado"
            text="Escolha uma reunião com registros.csv e acione Gerar para executar o script de relatório existente."
          />
        </section>
      )}
    </main>
  );
}

function App() {
  const page = useHashPage();
  const [status, setStatus] = useState(null);
  const [meetings, setMeetings] = useState([]);
  const [selectedName, setSelectedName] = useState("");
  const [loadError, setLoadError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [statusData, meetingsData] = await Promise.all([
        api("/api/status"),
        api("/api/reunioes"),
      ]);
      const nextMeetings = meetingsData.reunioes || [];
      setStatus(statusData);
      setMeetings(nextMeetings);
      setLoadError("");
      setSelectedName((current) => {
        if (current && nextMeetings.some((meeting) => meeting.nome === current)) {
          return current;
        }

        return nextMeetings[0]?.nome || "";
      });
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, page]);

  useEffect(() => {
    if (page !== "ao-vivo") {
      return undefined;
    }

    const interval = window.setInterval(refresh, POLL_INTERVAL);
    return () => window.clearInterval(interval);
  }, [page, refresh]);

  const selectedMeeting = useMemo(
    () => meetings.find((meeting) => meeting.nome === selectedName) || null,
    [meetings, selectedName]
  );

  const latest = status?.reuniao_mais_recente || meetings[0] || null;
  const PageComponent = {
    inicio: HomePage,
    "ao-vivo": LivePage,
    relatorio: ReportPage,
  }[page];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Mic size={21} />
          </div>
          <div>
            <strong>Sala Inteligente</strong>
            <span>IA de áudio local</span>
          </div>
        </div>
        <nav>
          {pages.map((item) => {
            const Icon = item.icon;
            return (
              <a className={page === item.id ? "active" : ""} href={`#${item.id}`} key={item.id}>
                <Icon size={18} />
                {item.label}
              </a>
            );
          })}
        </nav>
        <div className="sidebar-status">
          <span className={status?.processo?.rodando ? "dot live" : "dot"} />
          {status?.processo?.rodando ? "Capturando reunião" : "Aguardando captura"}
        </div>
      </aside>

      <div className="content">
        {loadError ? <div className="error top-error">{loadError}</div> : null}
        <PageComponent
          status={status}
          latest={latest}
          meetings={meetings}
          selectedName={selectedName}
          setSelectedName={setSelectedName}
          selectedMeeting={selectedMeeting}
          onRefresh={refresh}
        />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
