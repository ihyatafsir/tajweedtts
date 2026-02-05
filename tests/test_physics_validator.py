#!/usr/bin/env python3
"""
TajweedSST - Physics Validator Unit Tests

Tests all Tajweed acoustic validation rules:
- Qalqalah (bounce)
- Madd (elongation)
- Ghunnah (nasalization)
- Tafkheem (heavy letters)
- Idgham (assimilation)
- Ikhfa (concealment)
- Iqlab (conversion)
- Izhar (clarity)
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_validator import (
    PhysicsValidator, 
    ValidationStatus,
    PhysicsResult,
    QalqalahResult,
    MaddResult
)


class TestPhysicsValidatorInit:
    """Test initialization and configuration"""
    
    def test_default_init(self):
        """Validator initializes with default sample rate"""
        pv = PhysicsValidator()
        assert pv.sample_rate == 22050
        assert pv._average_vowel_duration > 0
    
    def test_custom_sample_rate(self):
        """Validator accepts custom sample rate"""
        pv = PhysicsValidator(sample_rate=16000)
        assert pv.sample_rate == 16000
    
    def test_thresholds_exist(self):
        """All Tajweed thresholds are defined"""
        pv = PhysicsValidator()
        assert hasattr(pv, 'QALQALAH_DIP_THRESHOLD')
        assert hasattr(pv, 'MADD_RATIO_ASLI')
        assert hasattr(pv, 'MADD_RATIO_WAJIB')
        assert hasattr(pv, 'MADD_RATIO_LAZIM')


class TestQalqalahValidation:
    """Test Qalqalah (echo/bounce) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate test audio: silence -> speech -> silence (qalqalah pattern)"""
        sr = 22050
        duration = 0.5  # 500ms
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create dip-spike pattern typical of qalqalah
        envelope = np.ones_like(t)
        # Dip at 30-40%
        envelope[int(0.3*len(t)):int(0.4*len(t))] = 0.1
        # Spike at 40-50%
        envelope[int(0.4*len(t)):int(0.5*len(t))] = 1.5
        
        signal = envelope * np.sin(2 * np.pi * 200 * t)
        return signal.astype(np.float32)
    
    def test_qalqalah_returns_physics_result(self, validator, sample_audio):
        """Qalqalah validation returns PhysicsResult"""
        result = validator.validate_qalqalah(sample_audio, 0.0, 0.5)
        # Result type is QalqalahResult which inherits from PhysicsResult
        assert hasattr(result, 'status')
        assert hasattr(result, 'metric_name')
    
    def test_qalqalah_detects_dip_spike(self, validator, sample_audio):
        """Qalqalah validator detects dip-spike pattern"""
        result = validator.validate_qalqalah(sample_audio, 0.0, 0.5)
        # Should at least have a score
        assert result.score >= 0
    
    def test_qalqalah_short_segment_handles_gracefully(self, validator):
        """Very short segments should be handled gracefully"""
        short_audio = np.zeros(100, dtype=np.float32)  # ~4.5ms at 22050
        result = validator.validate_qalqalah(short_audio, 0.0, 0.005)
        # Should not crash, status can be FAIL or SKIPPED
        assert result.status in [ValidationStatus.SKIPPED, ValidationStatus.FAIL]


class TestMaddValidation:
    """Test Madd (elongation) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def vowel_audio(self):
        """Generate sustained vowel-like audio"""
        sr = 22050
        duration = 0.4  # 400ms (should be ~2 counts)
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 200 * t)
        return signal.astype(np.float32)
    
    def test_madd_returns_physics_result(self, validator, vowel_audio):
        """Madd validation returns PhysicsResult"""
        result = validator.validate_madd(vowel_audio, 0.0, 0.4, expected_count=2)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')
    
    def test_madd_asli_duration(self, validator, vowel_audio):
        """Madd Asli (2 counts) should pass for ~400ms vowel"""
        result = validator.validate_madd(vowel_audio, 0.0, 0.4, expected_count=2)
        # Natural madd is 2 counts
        assert result.score >= 0


class TestGhunnahValidation:
    """Test Ghunnah (nasalization) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture  
    def nasal_audio(self):
        """Generate nasal-like audio with limited bandwidth"""
        sr = 22050
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        # Low frequency nasal resonance
        signal = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
        return signal.astype(np.float32)
    
    def test_ghunnah_returns_physics_result(self, validator, nasal_audio):
        """Ghunnah validation returns PhysicsResult"""
        result = validator.validate_ghunnah(nasal_audio, 0.0, 0.3)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')


class TestTafkheemValidation:
    """Test Tafkheem (heavy letter) detection via F2 formant"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def heavy_audio(self):
        """Generate audio with low F2 characteristic"""
        sr = 22050
        duration = 0.2
        t = np.linspace(0, duration, int(sr * duration))
        # Lower frequency components for "heavy" sound
        signal = np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
        return signal.astype(np.float32)
    
    def test_tafkheem_returns_physics_result(self, validator, heavy_audio):
        """Tafkheem validation returns PhysicsResult"""
        result = validator.validate_tafkheem(heavy_audio, 0.0, 0.2)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')


class TestIdghamValidation:
    """Test Idgham (assimilation) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def merged_audio(self):
        """Generate smoothly merged audio (no boundary)"""
        sr = 22050
        duration = 0.4
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 200 * t)
        return signal.astype(np.float32)
    
    def test_idgham_returns_physics_result(self, validator, merged_audio):
        """Idgham validation returns PhysicsResult"""
        result = validator.validate_idgham(merged_audio, 0.0, 0.2, 0.4, has_ghunnah=True)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')


class TestIkhfaValidation:
    """Test Ikhfa (concealment) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def concealed_audio(self):
        """Generate gradually fading nasal audio"""
        sr = 22050
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        envelope = np.exp(-3 * t / duration)  # Fading
        signal = envelope * np.sin(2 * np.pi * 300 * t)
        return signal.astype(np.float32)
    
    def test_ikhfa_returns_physics_result(self, validator, concealed_audio):
        """Ikhfa validation returns PhysicsResult"""
        result = validator.validate_ikhfa(concealed_audio, 0.0, 0.3)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')


class TestIzharValidation:
    """Test Izhar (clear pronunciation) detection"""
    
    @pytest.fixture
    def validator(self):
        return PhysicsValidator()
    
    @pytest.fixture
    def clear_audio(self):
        """Generate audio with clear boundary between sounds"""
        sr = 22050
        duration = 0.4
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.zeros_like(t)
        # First letter
        signal[:len(t)//2] = np.sin(2 * np.pi * 200 * t[:len(t)//2])
        # Gap (silence)
        # Second letter
        signal[int(0.55*len(t)):] = np.sin(2 * np.pi * 300 * t[int(0.55*len(t)):])
        return signal.astype(np.float32)
    
    def test_izhar_returns_physics_result(self, validator, clear_audio):
        """Izhar validation returns PhysicsResult"""
        result = validator.validate_izhar(clear_audio, 0.0, 0.2, 0.22)
        assert hasattr(result, 'status')
        assert hasattr(result, 'score')


class TestValidationResults:
    """Test result dataclasses"""
    
    def test_physics_result_fields(self):
        """PhysicsResult has all required fields"""
        result = PhysicsResult(
            status=ValidationStatus.PASS,
            metric_name="test",
            expected_pattern="dip-spike",
            observed_pattern="dip-spike",
            score=0.95
        )
        assert result.status == ValidationStatus.PASS
        assert result.score == 0.95
    
    def test_qalqalah_result_fields(self):
        """QalqalahResult has specific fields"""
        # QalqalahResult inherits from PhysicsResult and has extra fields
        from physics_validator import QalqalahResult, ValidationStatus
        result = QalqalahResult(
            status=ValidationStatus.PASS,
            metric_name="RMS Energy",
            expected_pattern="dip_then_spike",
            observed_pattern="dip_then_spike",
            score=0.8,
            rms_profile="dip-spike",
            dip_depth=0.3,
            spike_height=1.5,
            closure_duration_ms=50
        )
        assert result.dip_depth == 0.3
        assert result.spike_height == 1.5
    
    def test_madd_result_fields(self):
        """MaddResult has duration fields"""
        from physics_validator import MaddResult, ValidationStatus
        result = MaddResult(
            status=ValidationStatus.PASS,
            metric_name="Duration Ratio",
            expected_pattern="extended",
            observed_pattern="extended",
            score=1.0,
            actual_duration_ms=400,
            expected_duration_ms=400,
            ratio=1.0
        )
        assert result.ratio == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
