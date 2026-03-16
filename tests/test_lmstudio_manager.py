import unittest
from unittest.mock import Mock, patch

from src.modules.lmstudio_manager import LMStudioManager


class LMStudioManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = LMStudioManager()
        self.manager.model_poll_interval = 0
        self.manager.model_load_timeout = 1
        self.manager.unload_wait_seconds = 0
        self.manager.readiness_timeout = 1

    @patch("src.modules.lmstudio_manager.time.sleep", return_value=None)
    def test_load_model_unloads_existing_models_then_loads_with_preset(self, _sleep):
        desired_model = "qwen2.5-coder-7b-instruct"
        states = [
            ["deepseek-coder-6.7b-instruct"],
            [desired_model],
            [desired_model],
        ]
        command_calls = []

        def tracked_get_loaded_models():
            value = states[min(tracked_get_loaded_models.calls, len(states) - 1)]
            tracked_get_loaded_models.calls += 1
            return value

        tracked_get_loaded_models.calls = 0

        def fake_run_lms_command(command, check=True):
            command_calls.append((command, check))
            return True

        self.manager.get_loaded_models = tracked_get_loaded_models
        self.manager._run_lms_command = fake_run_lms_command
        self.manager.is_model_ready_for_generation = Mock(return_value=True)

        self.assertTrue(self.manager.load_model(desired_model))
        self.assertEqual(
            command_calls,
            [
                (["lms", "unload", "--all"], True),
                (["lms", "load", desired_model, "--preset", "Bringer_RAG"], True),
            ],
        )

    @patch("src.modules.lmstudio_manager.time.sleep", return_value=None)
    def test_load_model_uses_selected_model_without_unload_when_nothing_loaded(self, _sleep):
        desired_models = [
            "qwen2.5-coder-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-coder-1.5b-instruct",
        ]

        for desired_model in desired_models:
            with self.subTest(desired_model=desired_model):
                manager = LMStudioManager()
                manager.model_poll_interval = 0
                manager.model_load_timeout = 1
                manager.unload_wait_seconds = 0
                manager.readiness_timeout = 1

                states = [
                    [],
                    [desired_model],
                    [desired_model],
                ]
                command_calls = []

                def tracked_get_loaded_models():
                    value = states[min(tracked_get_loaded_models.calls, len(states) - 1)]
                    tracked_get_loaded_models.calls += 1
                    return value

                tracked_get_loaded_models.calls = 0

                def fake_run_lms_command(command, check=True):
                    command_calls.append((command, check))
                    return True

                manager.get_loaded_models = tracked_get_loaded_models
                manager._run_lms_command = fake_run_lms_command
                manager.is_model_ready_for_generation = Mock(return_value=True)

                self.assertTrue(manager.load_model(desired_model))
                self.assertEqual(
                    command_calls,
                    [(["lms", "load", desired_model, "--preset", "Bringer_RAG"], True)],
                )

    def test_ensure_ready_starts_server_when_not_running(self):
        manager = LMStudioManager()
        manager.is_server_running = Mock(return_value=False)
        manager.start_server = Mock()
        manager.load_model = Mock(return_value=True)

        manager.ensure_ready("qwen2.5-coder-7b-instruct")

        manager.start_server.assert_called_once()
        manager.load_model.assert_called_once_with("qwen2.5-coder-7b-instruct")

    @patch("src.modules.lmstudio_manager.httpx.Client")
    def test_get_loaded_models_uses_models_endpoint(self, mock_client_class):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": [{"id": "qwen2.5-coder-7b-instruct"}]}

        mock_client = Mock()
        mock_client.get.return_value = response
        mock_client_class.return_value.__enter__.return_value = mock_client

        loaded_models = self.manager.get_loaded_models()

        self.assertEqual(loaded_models, ["qwen2.5-coder-7b-instruct"])
        mock_client.get.assert_called_once_with(f"{self.manager.api_base}/models")

    @patch("src.modules.lmstudio_manager.subprocess.run")
    def test_stop_server_runs_stop_command_when_started_by_bringer(self, mock_run):
        self.manager.started_by_bringer = True

        self.manager.stop_server()

        mock_run.assert_called_once_with(
            ["lms", "server", "stop"],
            stdout=-3,
            stderr=-3,
            shell=False,
            check=False,
        )

    @patch("src.modules.lmstudio_manager.subprocess.run")
    def test_stop_server_skips_stop_when_server_was_not_started_by_bringer(self, mock_run):
        self.manager.started_by_bringer = False

        self.manager.stop_server()

        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
