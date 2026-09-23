from __future__ import annotations

import csv
import os
import platform
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MEETINGS_DIR = ROOT_DIR / "reunioes"
MODELS_DIR = ROOT_DIR / "modelos"

ANALYSIS_SCRIPT = SRC_DIR / "06_analisar_reuniao.py"
REPORT_SCRIPT = SRC_DIR / "07_gerar_relatorio.py"

MEETING_PATTERN = re.compile(r"reuniao_\d+")

app = FastAPI(title="Sala de Reuniao Inteligente API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/arquivos/reunioes",
    StaticFiles(directory=MEETINGS_DIR, check_dir=False),
    name="reunioes",
)

analysis_process: subprocess.Popen[str] | None = None
analysis_logs: list[str] = []
analysis_lock = threading.Lock()


def meeting_path(nome: str) -> Path:
    if not MEETING_PATTERN.fullmatch(nome):
        raise HTTPException(status_code=400, detail="Nome de reuniao invalido.")

    return MEETINGS_DIR / nome


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "trechos": 0,
            "duracao_segundos": 0,
            "participantes": [],
            "emocoes": {},
            "ultima_linha": None,
        }

    starts = [as_float(row.get("inicio")) for row in rows]
    ends = [as_float(row.get("fim")) for row in rows]
    participants = sorted(
        {
            str(row.get("pessoa", "")).strip()
            for row in rows
            if str(row.get("pessoa", "")).strip()
            and str(row.get("pessoa", "")).strip() != "desconhecido"
        }
    )
    emotions: dict[str, int] = {}

    for row in rows:
        emotion = str(row.get("emocao", "")).strip()
        if emotion:
            emotions[emotion] = emotions.get(emotion, 0) + 1

    return {
        "trechos": len(rows),
        "duracao_segundos": max(ends, default=0) - min(starts, default=0),
        "participantes": participants,
        "emocoes": emotions,
        "ultima_linha": rows[-1],
    }


def meeting_payload(path: Path) -> dict[str, Any]:
    csv_path = path / "registros.csv"
    rows = read_csv_rows(csv_path)
    report_path = path / "relatorio_reuniao.html"
    text_report_path = path / "relatorio.txt"
    graphs_dir = path / "graficos"
    low_confidence_path = graphs_dir / "baixa_confianca.csv"
    graphs = [
        graph.name
        for graph in sorted(graphs_dir.glob("*.png"))
    ] if graphs_dir.exists() else []

    modified_at = None
    if csv_path.exists():
        modified_at = datetime.fromtimestamp(
            csv_path.stat().st_mtime
        ).isoformat()

    return {
        "nome": path.name,
        "modificado_em": modified_at,
        "resumo": summarize_rows(rows),
        "registros": rows[-80:],
        "tem_csv": csv_path.exists(),
        "tem_relatorio": report_path.exists(),
        "tem_relatorio_txt": text_report_path.exists(),
        "tem_baixa_confianca": low_confidence_path.exists(),
        "registros_url": (
            f"/arquivos/reunioes/{path.name}/registros.csv"
            if csv_path.exists()
            else None
        ),
        "relatorio_url": (
            f"/arquivos/reunioes/{path.name}/relatorio_reuniao.html"
            if report_path.exists()
            else None
        ),
        "relatorio_txt_url": (
            f"/arquivos/reunioes/{path.name}/relatorio.txt"
            if text_report_path.exists()
            else None
        ),
        "relatorio_texto": read_text(text_report_path),
        "baixa_confianca_url": (
            f"/arquivos/reunioes/{path.name}/graficos/baixa_confianca.csv"
            if low_confidence_path.exists()
            else None
        ),
        "baixa_confianca": read_csv_rows(low_confidence_path),
        "graficos": [
            {
                "nome": graph,
                "url": f"/arquivos/reunioes/{path.name}/graficos/{graph}",
            }
            for graph in graphs
        ],
    }


def list_meeting_paths() -> list[Path]:
    if not MEETINGS_DIR.exists():
        return []

    paths = [
        path
        for path in MEETINGS_DIR.iterdir()
        if path.is_dir() and MEETING_PATTERN.fullmatch(path.name)
    ]

    return sorted(paths, key=lambda item: item.name, reverse=True)


def get_process_state() -> dict[str, Any]:
    global analysis_process

    with analysis_lock:
        process = analysis_process
        logs = analysis_logs[-120:]

    if process is None:
        return {
            "rodando": False,
            "codigo_saida": None,
            "logs": logs,
        }

    code = process.poll()
    if code is not None:
        with analysis_lock:
            analysis_process = None

    return {
        "rodando": code is None,
        "codigo_saida": code,
        "logs": logs,
    }


def capture_output(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return

    for line in process.stdout:
        clean_line = line.rstrip()
        with analysis_lock:
            analysis_logs.append(clean_line)
            del analysis_logs[:-200]


@app.get("/api/status")
def status() -> dict[str, Any]:
    meetings = [meeting_payload(path) for path in list_meeting_paths()]
    return {
        "scripts": {
            "analisar_reuniao": ANALYSIS_SCRIPT.exists(),
            "gerar_relatorio": REPORT_SCRIPT.exists(),
        },
        "modelos": {
            "pessoas": (MODELS_DIR / "modelo_pessoas.pkl").exists(),
            "emocoes": (MODELS_DIR / "modelo_emocoes.pkl").exists(),
        },
        "processo": get_process_state(),
        "reuniao_mais_recente": meetings[0] if meetings else None,
        "total_reunioes": len(meetings),
    }


@app.get("/api/reunioes")
def meetings() -> dict[str, Any]:
    return {
        "reunioes": [
            meeting_payload(path)
            for path in list_meeting_paths()
        ]
    }


@app.get("/api/reunioes/{nome}")
def meeting(nome: str) -> dict[str, Any]:
    path = meeting_path(nome)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reuniao nao encontrada.")

    return meeting_payload(path)


@app.post("/api/reunioes/analisar/iniciar")
def start_analysis() -> dict[str, Any]:
    global analysis_process

    if not ANALYSIS_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Script de analise nao encontrado.")

    with analysis_lock:
        if analysis_process is not None and analysis_process.poll() is None:
            raise HTTPException(status_code=409, detail="A analise ja esta em execucao.")

        analysis_logs.clear()

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        analysis_process = subprocess.Popen(
            [sys.executable, "-u", str(ANALYSIS_SCRIPT)],
            cwd=ROOT_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        threading.Thread(
            target=capture_output,
            args=(analysis_process,),
            daemon=True,
        ).start()

        if analysis_process.stdin is not None:
            analysis_process.stdin.write("\n")
            analysis_process.stdin.flush()

    return {"mensagem": "Analise iniciada.", "processo": get_process_state()}


@app.post("/api/reunioes/analisar/parar")
def stop_analysis() -> dict[str, Any]:
    global analysis_process

    with analysis_lock:
        process = analysis_process

    if process is None or process.poll() is not None:
        return {"mensagem": "Nenhuma analise em execucao.", "processo": get_process_state()}

    try:
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)

        process.wait(timeout=12)
    except Exception:
        process.terminate()

    return {"mensagem": "Analise encerrada.", "processo": get_process_state()}


@app.post("/api/reunioes/{nome}/relatorio")
def generate_report(nome: str) -> dict[str, Any]:
    path = meeting_path(nome)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reuniao nao encontrada.")

    if not (path / "registros.csv").exists():
        raise HTTPException(status_code=400, detail="A reuniao ainda nao possui registros.csv.")

    result = subprocess.run(
        [sys.executable, "-u", str(REPORT_SCRIPT), nome],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "mensagem": "Falha ao gerar relatorio.",
                "saida": result.stdout,
                "erro": result.stderr,
            },
        )

    payload = meeting_payload(path)
    payload["saida"] = result.stdout
    return payload
