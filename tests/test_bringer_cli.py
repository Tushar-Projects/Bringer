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
            "power_saver": False,
        }
        self.hardware_detector.select_profile.return_value = "high_performance"

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

    @patch("bringer_cli.get_config_manager")
    @patch("bringer_cli.print")
    @patch("bringer_cli.console.print")
    @patch("bringer_cli._load_runtime_modules")
    @patch("builtins.input", side_effect=["exit"])
    def test_default_mode_shows_minimal_startup_and_exit(
        self,
        _input,
        load_runtime_modules,
        console_print,
        _print,
        get_config_manager_mock,
    ):
        load_runtime_modules.return_value = self.runtime_modules
        mock_config = Mock()
        mock_config.get_active_mode.return_value = "auto"
        get_config_manager_mock.return_value = mock_config

        bringer_cli.launch_bringer([])

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Bringer AI Assistant\n", text_calls)
        self.assertTrue(any("Profile: high_performance" in str(args) for args in text_calls))
        self.assertIn("Ready.", text_calls)

    @patch("bringer_cli.console.print")
    def test_show_help_prints_clean_help_menu(self, console_print):
        bringer_cli.show_help()

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("[bold cyan]Bringer - Local AI Document Assistant[/bold cyan]\n", text_calls)
        self.assertIn("Usage:", text_calls)
        self.assertIn("  bringer                 Start the assistant", text_calls)
        self.assertIn("  bringer --debug         Run with detailed logs", text_calls)

    @patch("bringer_cli.console.print")
    @patch("src.modules.llama_manager.LlamaManager")
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
        lm_manager.current_model_path = "D:\\path\\to\\model.gguf"
        lm_manager_cls.return_value = lm_manager

        bringer_cli.run_status()

        text_calls = [args[0] for args, _ in console_print.call_args_list if args]
        self.assertIn("Bringer Status\n", text_calls)
        self.assertIn("Indexed files: 2", text_calls)
        self.assertIn("Total chunks: 4\n", text_calls)
        self.assertIn("Files:", text_calls)
        self.assertIn("- a.pdf", text_calls)

    def test_shutdown_bringer_stops_watcher_and_invokes_lm_shutdown(self):
        watcher = Mock()

        with patch("bringer_cli.console.print") as console_print, patch("src.modules.llama_manager.get_llama_manager") as get_llama_manager:
            bringer_cli.shutdown_bringer(watcher)

        watcher.stop.assert_called_once()
        get_llama_manager().shutdown.assert_called_once()
        console_print.assert_called_once_with("Shutting down Bringer...")

if __name__ == "__main__":
    unittest.main()
