"""Key loading. Every extra key is another rate-limit bucket, so the format
should never be the reason one silently goes unused."""

from __future__ import annotations

from fusion.config import load_api_key_entries, load_api_keys, mask_key


class TestSupportedForms:
    def test_a_single_unsuffixed_key(self):
        assert load_api_keys({"NIM_API_KEY": "nvapi-a"}) == ["nvapi-a"]

    def test_numbered_keys_in_suffix_order(self):
        env = {
            "NIM_API_KEY_2": "nvapi-b",
            "NIM_API_KEY_1": "nvapi-a",
            "NIM_API_KEY_3": "nvapi-c",
        }
        assert load_api_keys(env) == ["nvapi-a", "nvapi-b", "nvapi-c"]

    def test_double_digit_suffixes_sort_numerically(self):
        env = {f"NIM_API_KEY_{i}": f"nvapi-{i:02d}" for i in range(1, 13)}
        assert load_api_keys(env)[-1] == "nvapi-12"

    def test_a_comma_separated_list(self):
        assert load_api_keys({"NIM_API_KEYS": "nvapi-a,nvapi-b,nvapi-c"}) == [
            "nvapi-a",
            "nvapi-b",
            "nvapi-c",
        ]

    def test_a_list_separated_by_spaces_or_newlines(self):
        assert load_api_keys({"NIM_API_KEYS": "nvapi-a nvapi-b\nnvapi-c"}) == [
            "nvapi-a",
            "nvapi-b",
            "nvapi-c",
        ]

    def test_a_keys_file(self, tmp_path):
        path = tmp_path / "keys.txt"
        path.write_text("nvapi-a\nnvapi-b\n\nnvapi-c\n")

        assert load_api_keys({"NIM_API_KEYS_FILE": str(path)}) == [
            "nvapi-a",
            "nvapi-b",
            "nvapi-c",
        ]

    def test_a_missing_keys_file_is_a_warning_not_a_crash(self):
        assert load_api_keys({"NIM_API_KEYS_FILE": "/no/such/file"}) == []

    def test_forms_can_be_mixed(self, tmp_path):
        path = tmp_path / "keys.txt"
        path.write_text("nvapi-d\n")
        env = {
            "NIM_API_KEY": "nvapi-a",
            "NIM_API_KEY_1": "nvapi-b",
            "NIM_API_KEYS": "nvapi-c",
            "NIM_API_KEYS_FILE": str(path),
        }

        assert load_api_keys(env) == ["nvapi-a", "nvapi-b", "nvapi-c", "nvapi-d"]


class TestCleaning:
    def test_duplicates_collapse(self):
        env = {"NIM_API_KEY": "nvapi-a", "NIM_API_KEYS": "nvapi-a,nvapi-b"}

        assert load_api_keys(env) == ["nvapi-a", "nvapi-b"]

    def test_whitespace_and_quotes_are_stripped(self):
        env = {"NIM_API_KEY_1": '  "nvapi-a"  ', "NIM_API_KEY_2": "'nvapi-b'"}

        assert load_api_keys(env) == ["nvapi-a", "nvapi-b"]

    def test_the_shipped_placeholder_is_ignored(self):
        """.env.example ships `nvapi-...`; copying it must not yield a fake key."""
        env = {"NIM_API_KEY_1": "nvapi-...", "NIM_API_KEY_2": "nvapi-real"}

        assert load_api_keys(env) == ["nvapi-real"]

    def test_common_placeholder_words_are_ignored(self):
        env = {"NIM_API_KEY_1": "changeme", "NIM_API_KEY_2": "your-key-here"}

        assert load_api_keys(env) == []

    def test_blank_values_are_ignored(self):
        assert load_api_keys({"NIM_API_KEY_1": "", "NIM_API_KEY_2": "   "}) == []

    def test_unrelated_variables_are_ignored(self):
        env = {"NIM_API_KEY_X": "nope", "OTHER_KEY_1": "nope", "NIM_API_KEY": "yes-a"}

        assert load_api_keys(env) == ["yes-a"]


class TestOrigins:
    def test_each_key_records_where_it_came_from(self):
        env = {"NIM_API_KEY_1": "nvapi-a", "NIM_API_KEYS": "nvapi-b"}

        origins = [entry.origin for entry in load_api_key_entries(env)]

        assert origins == ["NIM_API_KEY_1", "NIM_API_KEYS"]


class TestMasking:
    def test_a_real_length_key_is_partly_visible(self):
        masked = mask_key("nvapi-abcdefghijklmnopqrstuvwxyz")

        assert masked.startswith("nvapi-abc")
        assert masked.endswith("wxyz")
        assert "defghijklmnop" not in masked

    def test_a_short_value_is_fully_hidden(self):
        assert mask_key("short") == "***"

    def test_different_keys_look_different(self):
        assert mask_key("nvapi-aaaaaaaaaaaaaaaaaaaa") != mask_key(
            "nvapi-bbbbbbbbbbbbbbbbbbbb"
        )
