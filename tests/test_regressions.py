"""Testes isolados: sem microfone, rede, servidor ou modelos reais."""
import contextlib
import importlib.util
import io
from pathlib import Path
import queue
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


class FakeApp:
    def __init__(self, **kwargs):
        pass

    def get(self, path):
        return lambda fn: fn

    post = get

    def add_middleware(self, *args, **kwargs):
        pass

    def mount(self, *args, **kwargs):
        pass


def external_stubs():
    modules = {name: Mock() for name in (
        "joblib", "features", "matplotlib", "matplotlib.pyplot",
        "sklearn", "sklearn.ensemble", "sklearn.model_selection", "sklearn.metrics",
        "fastapi.middleware", "fastapi.middleware.cors", "fastapi.staticfiles",
    )}
    modules["pyaudio"] = types.SimpleNamespace(paInt16=8, PyAudio=Mock())
    modules["fastapi"] = types.SimpleNamespace(FastAPI=FakeApp, HTTPException=HTTPException)
    return modules


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, external_stubs()):
        spec.loader.exec_module(module)
    return module


class Regressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_meetings_are_sorted_numerically(self):
        api = load("api/main.py")
        api.MEETINGS_DIR = self.base
        for name in ("reuniao_99", "reuniao_100", "reuniao_02", "outro"):
            (self.base / name).mkdir()
        self.assertEqual([p.name for p in api.list_meeting_paths()],
                         ["reuniao_100", "reuniao_99", "reuniao_02"])

    def test_exit_code_survives_multiple_status_polls(self):
        api = load("api/main.py")
        api.analysis_process = Mock()
        api.analysis_process.poll.return_value = 1
        self.assertEqual(api.get_process_state()["codigo_saida"], 1)
        self.assertEqual(api.get_process_state()["codigo_saida"], 1)

    def test_start_requires_both_models_without_launching(self):
        api = load("api/main.py")
        api.MODELS_DIR = self.base
        with patch.object(api.subprocess, "Popen") as launch:
            with self.assertRaises(HTTPException) as error:
                api.start_analysis()
            self.assertEqual(error.exception.status_code, 400)
            launch.assert_not_called()

    def test_stop_uses_stdin_without_killing_pending_work(self):
        api = load("api/main.py")
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        api.analysis_process = process
        result = api.stop_analysis()
        self.assertEqual(process.stdin.getvalue(), "parar\n")
        self.assertTrue(result["processo"]["rodando"])
        process.terminate.assert_not_called()
        process.send_signal.assert_not_called()

    def test_api_rejects_empty_report_without_subprocess(self):
        api = load("api/main.py")
        api.MEETINGS_DIR = self.base
        meeting = self.base / "reuniao_01"
        meeting.mkdir()
        (meeting / "registros.csv").write_text("inicio,fim,pessoa\n", encoding="utf-8")
        with patch.object(api.subprocess, "run") as run:
            with self.assertRaises(HTTPException) as error:
                api.generate_report("reuniao_01")
            self.assertEqual(error.exception.status_code, 400)
            run.assert_not_called()

    def test_api_report_timeout_is_explicit(self):
        api = load("api/main.py")
        api.MEETINGS_DIR = self.base
        meeting = self.base / "reuniao_01"
        meeting.mkdir()
        (meeting / "registros.csv").write_text("inicio,fim,pessoa\n0,4,joao\n", encoding="utf-8")
        with patch.object(api.subprocess, "run", side_effect=api.subprocess.TimeoutExpired("report", 180)):
            with self.assertRaises(HTTPException) as error:
                api.generate_report("reuniao_01")
            self.assertEqual(error.exception.status_code, 504)

    def test_report_script_returns_failure_for_missing_or_empty_csv(self):
        script = ROOT / "src/07_gerar_relatorio.py"
        meeting = self.base / "reunioes/reuniao_01"
        meeting.mkdir(parents=True)
        target = meeting / "registros.csv"
        for content in (None, "", "inicio,fim,pessoa,conf_pessoa,emocao,conf_emocao,arquivo\n"):
            with self.subTest(content=content):
                if content is not None:
                    target.write_text(content, encoding="utf-8")
                namespace = {"__file__": str(self.base / "src/report.py"), "__name__": "__main__"}
                with patch.dict(sys.modules, external_stubs()), patch.object(sys, "argv", [str(script), "reuniao_01"]):
                    with self.assertRaises(SystemExit) as error:
                        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
                    self.assertNotIn(error.exception.code, (None, 0))

    def test_training_rechecks_valid_counts(self):
        for script, classes, count in (
            ("03_treinar_pessoas.py", ["a", "b", "c", "d", "e"], 20),
            ("04_treinar_emocoes.py", ["alegre", "neutro", "triste", "irritado"], 25),
        ):
            with self.subTest(script=script):
                model = load("src/" + script)
                valid = np.repeat(classes, count)
                model.verificar_dataset(valid)
                with self.assertRaises(ValueError):
                    model.verificar_dataset(valid[1:])
                with self.assertRaises(ValueError):
                    model.verificar_dataset(np.repeat(classes[:2], count))

    def test_corrupt_files_cannot_satisfy_minimum(self):
        model = load("src/03_treinar_pessoas.py")
        model.PASTA_DATASET = str(self.base)
        for name in ("a", "b", "c", "d", "e"):
            folder = self.base / name
            folder.mkdir()
            for number in range(20):
                (folder / f"{number}.wav").touch()
        model.extrair_caracteristicas = Mock(side_effect=[np.zeros(64)] * 19 + [ValueError("invalid")])
        with self.assertRaisesRegex(ValueError, "19 áudios válidos"):
            model.carregar_dataset()

    def test_metadata_counts_only_valid_files(self):
        model = load("src/03_treinar_pessoas.py")
        model.PASTA_DATASET = str(self.base)
        for name in ("a", "b", "c", "d", "e"):
            folder = self.base / name
            folder.mkdir()
            for number in range(21):
                (folder / f"{number}.wav").touch()

        def extract(path):
            if Path(path).name == "0.wav":
                raise ValueError("invalid")
            return np.zeros(64)

        model.extrair_caracteristicas = extract
        x, y, counts = model.carregar_dataset()
        self.assertEqual(len(x), 100)
        self.assertEqual(counts, dict.fromkeys(("a", "b", "c", "d", "e"), 20))

    def test_capture_exact_blocks_and_partial_stop(self):
        capture = load("src/06_analisar_reuniao.py")
        stop, done = threading.Event(), threading.Event()
        pending, errors = queue.Queue(), []
        sizes = []

        def read(size, exception_on_overflow):
            self.assertTrue(exception_on_overflow)
            sizes.append(size)
            if sum(sizes) >= 64000 + 1024:
                stop.set()
            return b"\0\0" * size

        stream = Mock(read=read)
        capture.capturar_blocos(stream, stop, done, pending, errors, str(self.base), 2)
        self.assertTrue(done.is_set())
        self.assertFalse(errors)
        self.assertEqual(pending.qsize(), 2)
        first, last = pending.get(), pending.get()
        self.assertEqual(first[:2], (0.0, 4.0))
        self.assertEqual(last[:2], (4.0, 4.064))
        with wave.open(first[2], "rb") as wav:
            self.assertEqual(wav.getnframes(), 64000)

    def test_overflow_is_reported_and_capture_finishes(self):
        capture = load("src/06_analisar_reuniao.py")
        stop, done = threading.Event(), threading.Event()
        errors = []
        stream = Mock()
        stream.read.side_effect = OSError("overflow")
        capture.capturar_blocos(stream, stop, done, queue.Queue(), errors, str(self.base), 2)
        self.assertTrue(stop.is_set())
        self.assertTrue(done.is_set())
        self.assertEqual(str(errors[0]), "overflow")

    def test_stop_command_and_eof(self):
        capture = load("src/06_analisar_reuniao.py")
        for text in ("parar\n", ""):
            event = threading.Event()
            capture.aguardar_encerramento(io.StringIO(text), event)
            self.assertTrue(event.is_set())

    def test_main_drains_saved_blocks_after_stop(self):
        capture = load("src/06_analisar_reuniao.py")
        capture.PASTA_REUNIOES = str(self.base)
        capture.carregar_modelos = Mock(return_value=(object(), object()))
        capture.classificar_audio = Mock(return_value=("joao", 0.9, "neutro", 0.8))
        audio = capture.pyaudio.PyAudio.return_value
        audio.get_sample_size.return_value = 2

        def producer(stream, stop, done, pending, errors, folder, size):
            for number in (1, 2):
                path = capture.salvar_audio(b"\0\0" * 64000, number, folder, size)
                pending.put(((number - 1) * 4, number * 4, path))
            stop.set()
            done.set()

        with patch.object(sys, "argv", ["capture"]), patch("builtins.input", return_value=""), patch.object(capture, "capturar_blocos", side_effect=producer):
            self.assertEqual(capture.main(), 0)
        rows = pd.read_csv(self.base / "reuniao_01/registros.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows["fim"].tolist(), [4, 8])
        audio.open.return_value.close.assert_called_once()
        audio.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
