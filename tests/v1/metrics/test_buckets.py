# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for histogram bucket definitions."""

import pytest

from vllm.v1.metrics.buckets import (
    BUCKET_PROFILES,
    DEFAULT_BUCKETS,
    METRIC_BUCKET_MAPPING,
    BucketType,
    build_1_2_5_buckets,
    build_buckets,
    get_buckets,
    get_profile_buckets,
    merge_custom_buckets,
    validate_custom_buckets,
)

pytestmark = pytest.mark.cpu_test


class TestBucketDefinitions:
    """Tests for default bucket definitions."""

    def test_default_buckets_are_sorted(self):
        """Verify all default bucket definitions are in ascending order."""
        for bucket_type, buckets in DEFAULT_BUCKETS.items():
            assert buckets == tuple(sorted(buckets)), (
                f"{bucket_type.value} buckets are not sorted"
            )

    def test_default_buckets_have_no_duplicates(self):
        """Verify no duplicate values in bucket definitions."""
        for bucket_type, buckets in DEFAULT_BUCKETS.items():
            assert len(buckets) == len(set(buckets)), (
                f"{bucket_type.value} has duplicate bucket values"
            )

    def test_default_buckets_are_positive(self):
        """Verify all bucket values are positive."""
        for bucket_type, buckets in DEFAULT_BUCKETS.items():
            assert all(b > 0 for b in buckets), (
                f"{bucket_type.value} has non-positive bucket values"
            )

    def test_all_bucket_types_have_defaults(self):
        """Verify all BucketType enum values have default buckets defined."""
        dynamic_types = {BucketType.REQUEST_TOKEN_COUNT, BucketType.BATCH_SIZE}
        for bucket_type in BucketType:
            if bucket_type not in dynamic_types:
                # Dynamic types use build_1_2_5_buckets, others have defaults
                assert bucket_type in DEFAULT_BUCKETS, (
                    f"{bucket_type.value} missing from DEFAULT_BUCKETS"
                )


class TestBuildBuckets:
    """Tests for bucket building functions."""

    def test_build_buckets_basic(self):
        """Test basic bucket building with [1, 2, 5] mantissa."""
        result = build_buckets([1, 2, 5], 100)
        assert result == [1, 2, 5, 10, 20, 50, 100]

    def test_build_buckets_small_max(self):
        """Test bucket building with small max value."""
        result = build_buckets([1, 2, 5], 10)
        assert result == [1, 2, 5, 10]

    def test_build_buckets_very_small_max(self):
        """Test bucket building with very small max value."""
        result = build_buckets([1, 2, 5], 1)
        assert result == [1]

    def test_build_buckets_different_mantissa(self):
        """Test bucket building with different mantissa values."""
        result = build_buckets([1, 3], 30)
        assert result == [1, 3, 10, 30]

    def test_build_1_2_5_buckets(self):
        """Test the 1-2-5 bucket sequence helper."""
        assert build_1_2_5_buckets(100) == [1, 2, 5, 10, 20, 50, 100]
        assert build_1_2_5_buckets(10) == [1, 2, 5, 10]
        assert build_1_2_5_buckets(1) == [1]
        assert build_1_2_5_buckets(1000) == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    def test_build_buckets_empty_mantissa(self):
        """Test bucket building with empty mantissa list."""
        result = build_buckets([], 100)
        assert result == []

    def test_build_buckets_no_duplicates(self):
        """Test that build_buckets produces unique values."""
        result = build_buckets([1, 10], 100)
        assert len(result) == len(set(result))

    def test_build_buckets_handles_unsorted_mantissa(self):
        """Test bucket building with unsorted mantissa."""
        result = build_buckets([5, 1, 2], 100)
        assert result == sorted(result)

    def test_build_buckets_skips_non_positive(self):
        """Test that non-positive mantissa values are skipped."""
        result = build_buckets([0, 1, -1, 2], 10)
        assert result == [1, 2, 10]


class TestGetBuckets:
    """Tests for bucket retrieval function."""

    def test_get_buckets_static_types(self):
        """Test retrieval of static bucket types."""
        dynamic_types = {BucketType.REQUEST_TOKEN_COUNT, BucketType.BATCH_SIZE}
        for bucket_type in BucketType:
            if bucket_type in dynamic_types:
                continue
            buckets = get_buckets(bucket_type)
            assert buckets == DEFAULT_BUCKETS[bucket_type]

    def test_get_buckets_token_step_latency(self):
        """Test TOKEN_STEP_LATENCY buckets."""
        buckets = get_buckets(BucketType.TOKEN_STEP_LATENCY)
        # Should start at 10ms and extend to 80s
        assert buckets[0] == 0.01
        assert buckets[-1] == 80.0

    def test_get_buckets_prefill_latency(self):
        """Test PREFILL_LATENCY buckets."""
        buckets = get_buckets(BucketType.PREFILL_LATENCY)
        # Should start at 1ms and extend to 2560s
        assert buckets[0] == 0.001
        assert buckets[-1] == 2560.0

    def test_get_buckets_accumulated_phase_latency(self):
        """Test ACCUMULATED_PHASE_LATENCY buckets."""
        buckets = get_buckets(BucketType.ACCUMULATED_PHASE_LATENCY)
        # Should start at 100ms and extend to 7680s
        assert buckets[0] == 0.1
        assert buckets[-1] == 7680.0

    def test_get_buckets_cache_residency(self):
        """Test CACHE_RESIDENCY buckets."""
        buckets = get_buckets(BucketType.CACHE_RESIDENCY)
        # Should start at 1ms and extend to 3600s (1 hour)
        assert buckets[0] == 0.001
        assert buckets[-1] == 3600

    def test_get_buckets_batch_size_default(self):
        """Test BATCH_SIZE buckets with default max."""
        buckets = get_buckets(BucketType.BATCH_SIZE)
        # Should use 1-2-5 pattern up to default 16384
        assert buckets[0] == 1
        assert max(buckets) <= 16384

    def test_get_buckets_batch_size_dynamic(self):
        """Test BATCH_SIZE buckets with custom max_num_batched_tokens."""
        buckets = get_buckets(BucketType.BATCH_SIZE, max_num_batched_tokens=1000)
        assert buckets[0] == 1
        assert max(buckets) <= 1000
        # Should follow 1-2-5 pattern
        assert buckets == tuple(build_1_2_5_buckets(1000))

    def test_get_buckets_completion_count(self):
        """Test COMPLETION_COUNT buckets."""
        buckets = get_buckets(BucketType.COMPLETION_COUNT)
        assert buckets == (1, 2, 5, 10, 20)

    def test_get_buckets_request_token_count_requires_max_model_len(self):
        """Test REQUEST_TOKEN_COUNT requires max_model_len parameter."""
        with pytest.raises(ValueError, match="max_model_len is required"):
            get_buckets(BucketType.REQUEST_TOKEN_COUNT)

    def test_get_buckets_request_token_count_dynamic(self):
        """Test REQUEST_TOKEN_COUNT generates dynamic buckets."""
        buckets = get_buckets(BucketType.REQUEST_TOKEN_COUNT, max_model_len=1000)
        assert 1 in buckets
        assert max(buckets) <= 1000
        # Should follow 1-2-5 pattern
        assert buckets == tuple(build_1_2_5_buckets(1000))

    def test_get_buckets_request_token_count_various_lengths(self):
        """Test REQUEST_TOKEN_COUNT with various model lengths."""
        for max_len in [128, 512, 2048, 8192, 32768]:
            buckets = get_buckets(BucketType.REQUEST_TOKEN_COUNT, max_model_len=max_len)
            assert max(buckets) <= max_len
            assert min(buckets) == 1


class TestMetricBucketMapping:
    """Tests for metric to bucket type mapping."""

    def test_all_mapped_metrics_have_valid_bucket_types(self):
        """Ensure all mapped metrics reference valid bucket types."""
        for metric_name, bucket_type in METRIC_BUCKET_MAPPING.items():
            assert isinstance(bucket_type, BucketType), (
                f"Invalid bucket type for {metric_name}"
            )

    def test_time_metrics_use_time_buckets(self):
        """Verify time-based metrics use appropriate time bucket types."""
        time_bucket_types = {
            BucketType.TOKEN_STEP_LATENCY,
            BucketType.PREFILL_LATENCY,
            BucketType.ACCUMULATED_PHASE_LATENCY,
            BucketType.CACHE_RESIDENCY,
        }
        for metric_name, bucket_type in METRIC_BUCKET_MAPPING.items():
            if "seconds" in metric_name:
                assert bucket_type in time_bucket_types, (
                    f"{metric_name} should use a time-based bucket type"
                )

    def test_token_metrics_use_token_buckets(self):
        """Verify token-based metrics use REQUEST_TOKEN_COUNT."""
        token_metrics = [
            "vllm:request_prompt_tokens",
            "vllm:request_generation_tokens",
            "vllm:request_max_num_generation_tokens",
            "vllm:request_params_max_tokens",
            "vllm:request_prefill_kv_computed_tokens",
        ]
        for metric_name in token_metrics:
            assert METRIC_BUCKET_MAPPING[metric_name] == BucketType.REQUEST_TOKEN_COUNT

    def test_expected_metrics_are_mapped(self):
        """Verify expected metrics are present in the mapping."""
        expected_metrics = [
            "vllm:inter_token_latency_seconds",
            "vllm:time_to_first_token_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:request_prompt_tokens",
            "vllm:iteration_tokens_total",
            "vllm:request_params_n",
        ]
        for metric_name in expected_metrics:
            assert metric_name in METRIC_BUCKET_MAPPING, (
                f"{metric_name} missing from METRIC_BUCKET_MAPPING"
            )


class TestBucketProfiles:
    """Tests for histogram bucket profiles."""

    def test_get_profile_buckets_standard(self):
        """Test that standard profile returns default buckets."""
        profile_buckets = get_profile_buckets("standard")
        assert profile_buckets == DEFAULT_BUCKETS

    def test_get_profile_buckets_default_is_standard(self):
        """Test that default profile is standard."""
        profile_buckets = get_profile_buckets()
        assert profile_buckets == DEFAULT_BUCKETS

    def test_get_profile_buckets_invalid_profile(self):
        """Test that invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown histogram profile"):
            get_profile_buckets("nonexistent")

    def test_bucket_profiles_dict_has_standard(self):
        """Test that BUCKET_PROFILES contains standard profile."""
        assert "standard" in BUCKET_PROFILES
        assert BUCKET_PROFILES["standard"] == DEFAULT_BUCKETS

    def test_all_profiles_have_all_static_bucket_types(self):
        """Verify all profiles define all static bucket types."""
        static_types = [bt for bt in BucketType if bt != BucketType.REQUEST_TOKEN_COUNT]
        for profile_name, buckets in BUCKET_PROFILES.items():
            for bucket_type in static_types:
                assert bucket_type in buckets, (
                    f"Profile '{profile_name}' missing bucket type {bucket_type.value}"
                )

    def test_all_expected_profiles_exist(self):
        """Verify all expected profiles are defined."""
        expected_profiles = ["standard", "low-latency", "high-throughput", "batch"]
        for profile in expected_profiles:
            assert profile in BUCKET_PROFILES, f"Profile '{profile}' not found"

    def test_get_profile_buckets_all_profiles(self):
        """Test that all profiles can be retrieved."""
        for profile in ["standard", "low-latency", "high-throughput", "batch"]:
            buckets = get_profile_buckets(profile)
            assert isinstance(buckets, dict)
            assert len(buckets) > 0

    def test_profile_buckets_are_sorted(self):
        """Verify all profile bucket values are sorted."""
        for profile_name, buckets in BUCKET_PROFILES.items():
            for bucket_type, values in buckets.items():
                assert values == tuple(sorted(values)), (
                    f"Profile '{profile_name}' {bucket_type.value} buckets not sorted"
                )

    def test_profile_buckets_are_positive(self):
        """Verify all profile bucket values are positive."""
        for profile_name, buckets in BUCKET_PROFILES.items():
            for bucket_type, values in buckets.items():
                assert all(v > 0 for v in values), (
                    f"'{profile_name}' {bucket_type.value} has non-positive values"
                )


class TestCustomBuckets:
    """Tests for custom bucket validation and merging (PR 4)."""

    def test_validate_custom_buckets_valid(self):
        """Test validation of valid custom buckets."""
        custom = {"token_step_latency": [0.01, 0.05, 0.1, 0.5, 1.0]}
        result = validate_custom_buckets(custom)
        assert BucketType.TOKEN_STEP_LATENCY in result
        assert result[BucketType.TOKEN_STEP_LATENCY] == (0.01, 0.05, 0.1, 0.5, 1.0)

    def test_validate_custom_buckets_invalid_type(self):
        """Test validation rejects invalid bucket type."""
        custom = {"invalid_type": [0.1, 0.5, 1.0]}
        with pytest.raises(ValueError, match="Invalid bucket type"):
            validate_custom_buckets(custom)

    def test_validate_custom_buckets_empty_values(self):
        """Test validation rejects empty bucket values."""
        custom: dict[str, list[float]] = {"token_step_latency": []}
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_custom_buckets(custom)

    def test_validate_custom_buckets_non_positive(self):
        """Test validation rejects non-positive values."""
        custom = {"token_step_latency": [0.0, 0.1, 0.5]}
        with pytest.raises(ValueError, match="must be positive"):
            validate_custom_buckets(custom)

    def test_validate_custom_buckets_not_sorted(self):
        """Test validation rejects unsorted values."""
        custom = {"token_step_latency": [0.5, 0.1, 1.0]}
        with pytest.raises(ValueError, match="must be sorted"):
            validate_custom_buckets(custom)

    def test_merge_custom_buckets_no_overrides(self):
        """Test merge without overrides returns base profile."""
        result = merge_custom_buckets("standard", None)
        assert result == DEFAULT_BUCKETS

    def test_merge_custom_buckets_with_overrides(self):
        """Test merge applies overrides to base profile."""
        custom = {BucketType.TOKEN_STEP_LATENCY: (0.01, 0.1, 1.0)}
        result = merge_custom_buckets("standard", custom)
        assert result[BucketType.TOKEN_STEP_LATENCY] == (0.01, 0.1, 1.0)
        # Other bucket types should remain from base profile
        assert (
            result[BucketType.PREFILL_LATENCY]
            == DEFAULT_BUCKETS[BucketType.PREFILL_LATENCY]
        )

    def test_merge_custom_buckets_different_profile(self):
        """Test merge with different base profile."""
        custom = {BucketType.TOKEN_STEP_LATENCY: (0.01, 0.1, 1.0)}
        result = merge_custom_buckets("low-latency", custom)
        assert result[BucketType.TOKEN_STEP_LATENCY] == (0.01, 0.1, 1.0)
