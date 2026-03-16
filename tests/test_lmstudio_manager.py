import unittest
from unittest.mock import Mock, patch

from src.modules.lmstudio_manager import LMStudioManager


class LMStudioManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = LMStudioManager()
        self.manager.model_poll_interval = 0
        self.manager.model_load_timeout = 1
        self.manager.unload_timeout = 1
        self.manager.readiness_timeout = 1

    @patch("src.modules.lmstudio_manager.time.sleep", return_value=None)
    def test_load_model_unloads_other_models_before_loading_target(self, _sleep):
        desired_model = "qwen2.5-7b-instruct"
        states = [
            ["qwen2.5-coder-7b-instruct", "deepseek-coder-6.7b-instruct"],
            ["qwen2.5-coder-7b-instruct"],
            [],
            [],
            [desired_model],
            [desired_model],
        ]
        call_order = []
        call_order_states = []

        def tracked_get_loaded_models():
            value = states[min(len(call_order_states), len(states) - 1)]
            call_order_states.append(list(value))
            return value

        def fake_run_lms_command(command, background=False):
            call_order.append(("command", command, background))
            return True

        def fake_ready(model_name):
            call_order.append(("ready", model_name))
            return True

        self.manager.get_loaded_models = tracked_get_loaded_models
        self.manager._run_lms_command = fake_run_lms_command
        self.manager.is_model_ready_for_generation = fake_ready

        self.assertTrue(self.manager.load_model(desired_model))
        self.assertEqual(call_order[0], ("command", ["lms", "unload", "--all"], False))
        self.assertEqual(
            call_order[1],
            ("command", ["lms", "load", desired_model, "--preset", "Bringer_RAG"], True),
        )
        self.assertEqual(call_order[2], ("ready", desired_model))
        self.assertNotIn(desired_model, call_order_states[0])

    @patch("src.modules.lmstudio_manager.time.sleep", return_value=None)
    def test_load_model_with_no_models_still_loads_target(self, _sleep):
        desired_model = "qwen2.5-7b-instruct"
        states = [
            [],
            [],
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

        def fake_run_lms_command(command, background=False):
            command_calls.append((command, background))
            return True

        self.manager.get_loaded_models = tracked_get_loaded_models
        self.manager._run_lms_command = fake_run_lms_command
        self.manager.is_model_ready_for_generation = Mock(return_value=True)

        self.assertTrue(self.manager.load_model(desired_model))
        self.assertEqual(command_calls[0], (["lms", "unload", "--all"], False))
        self.assertEqual(
            command_calls[1],
            (["lms", "load", desired_model, "--preset", "Bringer_RAG"], True),
        )

    def test_ensure_ready_starts_server_when_not_running(self):
        manager = LMStudioManager()
        manager.is_server_running = Mock(side_effect=[False, True])
        manager.start_server = Mock()
        manager.load_model = Mock(return_value=True)

        manager.ensure_ready("qwen2.5-7b-instruct")

        manager.start_server.assert_called_once()
        manager.load_model.assert_called_once_with("qwen2.5-7b-instruct")

    @patch("src.modules.lmstudio_manager.httpx.Client")
    def test_get_loaded_models_uses_models_endpoint(self, mock_client_class):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": [{"id": "qwen2.5-7b-instruct"}]}

        mock_client = Mock()
        mock_client.get.return_value = response
        mock_client_class.return_value.__enter__.return_value = mock_client

        loaded_models = self.manager.get_loaded_models()

        self.assertEqual(loaded_models, ["qwen2.5-7b-instruct"])
        mock_client.get.assert_called_once_with(f"{self.manager.api_base}/models")


if __name__ == "__main__":
    unittest.main()
