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
        self.hardware_detector.select_model.return_value = "Qwen2.5-7B-Instruct-1M-Q6_K"

        self.lm_manager = Mock()
        self.watcher = Mock()
        self.pipeline = Mock()
        self.pipeline.run_rag.return_value = iter(("Answer text", "\n\nSources\n", "- doc.txt\n"))
        self.runtime_modules = (
            lambda: self.watcher,
            lambda: self.hardware_detector,
            lambda: self.lm_manager,
            lambda: self.pipeline,
        )

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["exit"])
    def test_default_mode_shows_minimal_startup_and_exit(
        self,
        _input,
        load_runtime_modules,
        _check_lms,
        console_print,
        _print,
    ):
        load_runtime_modules.return_value = self.runtime_modules

        bringer_cli.launch_bringer([])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Bringer AI Assistant\n", text_calls)
        self.assertIn("Model: Qwen2.5-7B-Instruct-1M-Q6_K (GPU)", text_calls)
        self.assertIn("Ready.", text_calls)
        self.assertIn("Shutting down Bringer...", text_calls)
        self.assertNotIn("Documents indexed: 71\n", text_calls)
        self.assertNotIn("Ask a question or type 'exit'.", text_calls)
        self.assertNotIn("[bold cyan]Answer:[/bold cyan] ", text_calls)

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["question", "exit"])
    def test_default_mode_streams_answer_without_answer_label(
        self,
        _input,
        load_runtime_modules,
        _check_lms,
        console_print,
        print_mock,
    ):
        load_runtime_modules.return_value = self.runtime_modules

        bringer_cli.launch_bringer([])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertNotIn("[bold cyan]Answer:[/bold cyan] ", text_calls)
        print_mock.assert_any_call("Answer text", end="", flush=True)
        print_mock.assert_any_call("\n\nSources\n", end="", flush=True)
        print_mock.assert_any_call("- doc.txt\n", end="", flush=True)

    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["exit"])
    def test_debug_mode_keeps_verbose_banner(
        self,
        _input,
        load_runtime_modules,
        _check_lms,
        console_print,
        _print,
    ):
        load_runtime_modules.return_value = self.runtime_modules

        bringer_cli.launch_bringer(["--debug"])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("\n[bold magenta]--- Bringer RAG Assistant ---[/bold magenta]\n", text_calls)
        self.assertTrue(any("RAG Assistant Ready" in text for text in text_calls))
        self.assertIn("Ask a question (type 'exit' or 'quit' to close)", text_calls)

    @patch("bringer_cli._check_lmstudio_cli", return_value=False)
    @patch("bringer_cli._load_runtime_modules")
    def test_launch_bringer_stops_early_when_lmstudio_cli_is_missing(self, load_runtime_modules, _check_lms):
        load_runtime_modules.return_value = self.runtime_modules

        bringer_cli.launch_bringer([])

        self.lm_manager.ensure_ready.assert_not_called()

    @patch("bringer_cli._load_runtime_modules", return_value=None)
    def test_launch_bringer_stops_early_when_runtime_imports_fail(self, _load_runtime_modules):
        bringer_cli.launch_bringer([])

        self.lm_manager.ensure_ready.assert_not_called()

    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["exit"])
    def test_exit_triggers_shutdown(self, _input, load_runtime_modules, _check_lms):
        load_runtime_modules.return_value = self.runtime_modules

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        self.lm_manager.ensure_ready.assert_called_once_with("Qwen2.5-7B-Instruct-1M-Q6_K")
        self.watcher.start.assert_called_once()
        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["quit"])
    def test_quit_triggers_shutdown(self, _input, load_runtime_modules, _check_lms):
        load_runtime_modules.return_value = self.runtime_modules

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    @patch("bringer_cli._check_lmstudio_cli", return_value=True)
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_ctrl_c_triggers_shutdown(self, _input, load_runtime_modules, _check_lms):
        load_runtime_modules.return_value = self.runtime_modules

        with patch("bringer_cli.shutdown_bringer") as shutdown_bringer:
            bringer_cli.launch_bringer([])

        shutdown_bringer.assert_called_once_with(self.watcher, self.lm_manager)

    @patch("bringer_cli.console.print")
    @patch("src.modules.hybrid_retriever.HybridRetriever")
    @patch("src.modules.vector_store.VectorStore")
    @patch("bringer_cli.Path")
    def test_run_reindex_mode_clears_db_reindexes_supported_files_and_rebuilds_bm25(
        self,
        path_cls,
        vector_store_cls,
        hybrid_retriever_cls,
        console_print,
    ):
        store = Mock()
        vector_store_cls.return_value = store
        hybrid_retriever = Mock()
        hybrid_retriever_cls.return_value = hybrid_retriever

        file1 = Mock()
        file1.is_file.return_value = True
        file1.suffix = ".pdf"
        file1.name = "file1.pdf"

        file2 = Mock()
        file2.is_file.return_value = True
        file2.suffix = ".docx"
        file2.name = "file2.docx"

        ignored = Mock()
        ignored.is_file.return_value = True
        ignored.suffix = ".exe"
        ignored.name = "ignore.exe"

        docs_path = Mock()
        docs_path.iterdir.return_value = [file1, file2, ignored]
        path_cls.return_value = docs_path

        bringer_cli.run_reindex_mode()

        store.clear.assert_called_once()
        store.process_file.assert_any_call(file1)
        store.process_file.assert_any_call(file2)
        self.assertEqual(store.process_file.call_count, 2)
        hybrid_retriever.rebuild_bm25_index.assert_called_once()
        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Reindexing documents...", text_calls)
        self.assertIn("Indexing file1.pdf...", text_calls)
        self.assertIn("Indexing file2.docx...", text_calls)
        self.assertIn("Reindex complete. 2 files processed.", text_calls)

    @patch("bringer_cli.console.print")
    @patch("src.modules.lmstudio_manager.LMStudioManager")
    @patch("src.modules.vector_store.VectorStore")
    def test_run_status_prints_index_and_model_status(self, vector_store_cls, lm_manager_cls, console_print):
        store = Mock()
        store.collection.count.return_value = 4
        store.collection.get.return_value = {
            "metadatas": [
                {"source_file": "a.pdf"},
                {"source_file": "b.docx"},
                {"source_file": "a.pdf"},
            ]
        }
        vector_store_cls.return_value = store

        lm_manager = Mock()
        lm_manager.get_loaded_models.return_value = ["Qwen2.5-7B-Instruct-1M-Q6_K"]
        lm_manager_cls.return_value = lm_manager

        bringer_cli.run_status()

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Bringer Status\n", text_calls)
        self.assertIn("Indexed files: 2", text_calls)
        self.assertIn("Total chunks: 4\n", text_calls)
        self.assertIn("Files:", text_calls)
        self.assertIn("- a.pdf", text_calls)
        self.assertIn("- b.docx", text_calls)
        self.assertIn("\nActive model:", text_calls)
        self.assertIn("- Qwen2.5-7B-Instruct-1M-Q6_K", text_calls)

    @patch("bringer_cli.run_status")
    def test_status_flag_runs_status_mode_and_exits(self, run_status):
        bringer_cli.launch_bringer(["--status"])
        run_status.assert_called_once()

    @patch("bringer_cli.console.print")
    def test_show_help_prints_clean_help_menu(self, console_print):
        bringer_cli.show_help()

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("[bold cyan]Bringer - Local AI Document Assistant[/bold cyan]\n", text_calls)
        self.assertIn("Usage:", text_calls)
        self.assertIn("  Bringer                 Start the assistant", text_calls)
        self.assertIn("  Bringer --debug         Run with detailed logs", text_calls)
        self.assertIn("  Bringer --status        Show indexed files and system status", text_calls)
        self.assertIn("  Bringer --reindex       Rebuild the document index", text_calls)
        self.assertIn("  Bringer --help          Show this help message\n", text_calls)
        self.assertIn("Description:", text_calls)
        self.assertIn("Examples:", text_calls)

    @patch("bringer_cli.show_help")
    @patch("bringer_cli._load_runtime_modules")
    def test_help_flag_runs_help_mode_and_exits(self, load_runtime_modules, show_help):
        bringer_cli.launch_bringer(["--help"])

        show_help.assert_called_once()
        load_runtime_modules.assert_not_called()

    @patch("bringer_cli.show_help")
    @patch("bringer_cli._load_runtime_modules")
    def test_short_help_flag_runs_help_mode_and_exits(self, load_runtime_modules, show_help):
        bringer_cli.launch_bringer(["-h"])

        show_help.assert_called_once()
        load_runtime_modules.assert_not_called()

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
