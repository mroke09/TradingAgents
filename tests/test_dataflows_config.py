"""Config isolation: get/set must not leak nested-dict references."""

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import get_config, run_with_config, set_config, use_config


@pytest.mark.unit
class DataflowsConfigIsolationTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_get_config_returns_deep_copy(self):
        cfg = get_config()
        cfg["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        cfg["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "yfinance")
        self.assertNotIn("get_stock_data", fresh["tool_vendors"])

    def test_set_config_does_not_alias_caller_nested_dicts(self):
        custom = copy.deepcopy(default_config.DEFAULT_CONFIG)
        custom["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        custom["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        set_config(custom)

        custom["data_vendors"]["core_stock_apis"] = "yfinance"
        custom["tool_vendors"]["get_stock_data"] = "yfinance"

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")

    def test_partial_nested_update_preserves_existing_defaults(self):
        set_config(
            {
                "data_vendors": {
                    "core_stock_apis": "alpha_vantage",
                }
            }
        )

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        self.assertEqual(fresh["data_vendors"]["technical_indicators"], "yfinance")
        self.assertEqual(fresh["data_vendors"]["fundamental_data"], "yfinance")
        self.assertEqual(fresh["data_vendors"]["news_data"], "yfinance")

    def test_nested_dict_updates_merge_one_level_deep(self):
        set_config({"tool_vendors": {"get_stock_data": "alpha_vantage"}})
        set_config({"tool_vendors": {"get_news": "alpha_vantage"}})

        fresh = get_config()
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_news"], "alpha_vantage")

    def test_use_config_is_context_local(self):
        local_config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        local_config["data_vendors"]["core_stock_apis"] = "alpha_vantage"

        with use_config(local_config):
            self.assertEqual(
                get_config()["data_vendors"]["core_stock_apis"],
                "alpha_vantage",
            )

        self.assertEqual(get_config()["data_vendors"]["core_stock_apis"], "yfinance")

    def test_run_with_config_propagates_to_worker_thread(self):
        local_config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        local_config["output_language"] = "中文"

        def read_language():
            return get_config()["output_language"]

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                run_with_config,
                local_config,
                read_language,
            ).result()

        self.assertEqual(result, "中文")
        self.assertEqual(get_config()["output_language"], "English")
