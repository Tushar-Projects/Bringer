import unittest
from unittest.mock import Mock, patch

import bringer_cli


class BringerCliTests(unittest.TestCase):
    def setUp(self):
        self.hardware_detector = Mock()
        self.hardware_detector.detect_hardware.return_value = {
            "gpu_available": True,
            "gpu_name": "NVIDIA RTX 4070 Laptop GPU",
            "plugged_in": True,
        }
        self.hardware_detector.select_model.return_value = "qwen2.5-coder-7b-instruct"

        self.lm_manager = Mock()
        self.watcher = Mock()
        self.pipeline = Mock()
        self.pipeline.run_rag.return_value = iter(("Answer text", "\n\nSources\n", "- doc.txt\n"))
        self.vector_store = Mock()
        self.vector_store.get_stats.return_value = {"total_documents": 71}

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=["exit"])
    def test_default_mode_shows_clean_startup_and_exit(
        self,
        _input,
        hardware_detector_cls,
        lm_manager_cls,
        watcher_cls,
        pipeline_cls,
        vector_store_cls,
        console_print,
        _print,
    ):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        bringer_cli.launch_bringer([])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Bringer AI Assistant\n", text_calls)
        self.assertIn("Model: qwen2.5-coder-7b-instruct (GPU)", text_calls)
        self.assertIn("Documents indexed: 71\n", text_calls)
        self.assertIn("Ready.", text_calls)
        self.assertIn("Ask a question or type 'exit'.", text_calls)
        self.assertIn("Shutting down Bringer...", text_calls)
        self.assertNotIn("[dim]Detecting hardware...[/dim]", text_calls)
        self.assertNotIn("[bold cyan]Answer:[/bold cyan] ", text_calls)

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=["question", "exit"])
    def test_default_mode_streams_answer_without_answer_label(
        self,
        _input,
        hardware_detector_cls,
        lm_manager_cls,
        watcher_cls,
        pipeline_cls,
        vector_store_cls,
        console_print,
        print_mock,
    ):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        bringer_cli.launch_bringer([])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertNotIn("[bold cyan]Answer:[/bold cyan] ", text_calls)
        print_mock.assert_any_call("Answer text", end="", flush=True)
        print_mock.assert_any_call("\n\nSources\n", end="", flush=True)
        print_mock.assert_any_call("- doc.txt\n", end="", flush=True)

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=["exit"])
    def test_debug_mode_keeps_verbose_banner(
        self,
        _input,
        hardware_detector_cls,
        lm_manager_cls,
        watcher_cls,
        pipeline_cls,
        vector_store_cls,
        console_print,
        _print,
    ):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        bringer_cli.launch_bringer(["--debug"])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("\n[bold magenta]--- Bringer RAG Assistant ---[/bold magenta]\n", text_calls)
        self.assertTrue(any("RAG Assistant Ready" in text for text in text_calls))
        self.assertIn("Ask a question (type 'exit' or 'quit' to close)", text_calls)

    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=["exit"])
    def test_exit_triggers_shutdown(self, _input, hardware_detector_cls, lm_manager_cls, watcher_cls, pipeline_cls, vector_store_cls):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        self.lm_manager.ensure_ready.assert_called_once_with("qwen2.5-coder-7b-instruct")
        self.watcher.start.assert_called_once()
        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=["quit"])
    def test_quit_triggers_shutdown(self, _input, hardware_detector_cls, lm_manager_cls, watcher_cls, pipeline_cls, vector_store_cls):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    @patch("bringer_cli.VectorStore")
    @patch("bringer_cli.RAGPipeline")
    @patch("bringer_cli.DocumentWatcher")
    @patch("bringer_cli.LMStudioManager")
    @patch("bringer_cli.HardwareDetector")
    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_ctrl_c_triggers_shutdown(self, _input, hardware_detector_cls, lm_manager_cls, watcher_cls, pipeline_cls, vector_store_cls):
        hardware_detector_cls.return_value = self.hardware_detector
        lm_manager_cls.return_value = self.lm_manager
        watcher_cls.return_value = self.watcher
        pipeline_cls.return_value = self.pipeline
        vector_store_cls.return_value = self.vector_store

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    def test_shutdown_bringer_stops_watcher_and_invokes_lm_shutdown(self):
        watcher = Mock()
        lm_manager = Mock()

        with patch("bringer_cli.console.print") as console_print, patch("bringer_cli.shutdown_lmstudio") as shutdown_lmstudio:
            bringer_cli.shutdown_bringer(watcher, lm_manager)

        watcher.stop.assert_called_once()
        shutdown_lmstudio.assert_called_once_with(lm_manager)
        console_print.assert_called_once_with("Shutting down Bringer...")


if __name__ == "__main__":
    unittest.main()
