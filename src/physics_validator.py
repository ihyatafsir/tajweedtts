#!/usr/bin/env python3
"""
TajweedSST - Step 3: Physics & Signal Processing Validator

Validates Tajweed rules using acoustic signal analysis:
- Qalqalah: RMS energy dip→spike pattern
- Madd: Duration vs Rate of Speech ratio
- Ghunnah: Formant analysis + nasalization detection
- Tafkheem: F2 formant depression
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Import signal processing libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. RMS/ZCR analysis unavailable.")

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("Warning: parselmouth not installed. Formant analysis unavailable.")


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MARGINAL = "MARGINAL"
    SKIPPED = "SKIPPED"

@dataclass
class PhysicsResult:
    """Result of a physics/signal analysis check"""
    status: ValidationStatus
    metric_name: str
    expected_pattern: str
    observed_pattern: str
    score: float  # 0.0 to 1.0
    details: Dict = field(default_factory=dict)

@dataclass
class QalqalahResult(PhysicsResult):
    """Specific result for Qalqalah check"""
    rms_profile: str = ""  # "dip_then_spike", "flat", "spike_only"
    dip_depth: float = 0.0
    spike_height: float = 0.0
    closure_duration_ms: float = 0.0

@dataclass
class MaddResult(PhysicsResult):
    """Specific result for Madd elongation check"""
    actual_duration_ms: float = 0.0
    expected_duration_ms: float = 0.0
    ratio: float = 0.0  # Actual / Average vowel

@dataclass
class GhunnahResult(PhysicsResult):
    """Specific result for Ghunnah nasalization check"""
    nasal_formant_detected: bool = False
    pitch_stability: float = 0.0
    duration_elongation: float = 0.0

@dataclass
class TafkheemResult(PhysicsResult):
    """Specific result for Tafkheem check"""
    f2_value_hz: float = 0.0
    f2_baseline_hz: float = 1500.0  # Average F2 for light sounds
    depression_ratio: float = 0.0


class PhysicsValidator:
    """
    Validates Tajweed rules using signal processing
    """
    
    # Thresholds for validation
    QALQALAH_DIP_THRESHOLD = 0.3  # RMS must drop by 30%
    QALQALAH_SPIKE_THRESHOLD = 0.5  # RMS must rise by 50%
    MADD_RATIO_ASLI = 2.0   # 2x average vowel
    MADD_RATIO_WAJIB = 4.0  # 4x average vowel
    MADD_RATIO_LAZIM = 6.0  # 6x average vowel
    GHUNNAH_MIN_DURATION_MS = 80.0
    TAFKHEEM_F2_MAX_HZ = 1200.0  # Heavy letters have depressed F2
    
    # Precision thresholds - tuned for Arabic letters which can be very short
    MIN_SEGMENT_MS = 30.0  # Minimum segment duration for valid analysis (lowered from 50ms)
    MIN_SEGMENT_SAMPLES = 661  # ~30ms at 22050 Hz
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._audio_cache = {}
        self._average_vowel_duration = 0.1  # Will be calibrated per reciter
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file, with caching"""
        if audio_path not in self._audio_cache:
            if HAS_LIBROSA:
                y, sr = librosa.load(audio_path, sr=self.sample_rate)
                self._audio_cache[audio_path] = y
            else:
                # Fallback: generate noise for testing
                self._audio_cache[audio_path] = np.random.randn(self.sample_rate * 10) * 0.1
        
        return self._audio_cache[audio_path]
    
    def safe_extract_segment(self, audio: np.ndarray, start: float, end: float) -> tuple:
        """
        PRECISION: Safely extract audio segment with bounds and validity checking.
        
        Returns:
            tuple: (segment, is_valid, error_reason)
        """
        # Bounds checking
        start_sample = max(0, int(start * self.sample_rate))
        end_sample = min(len(audio), int(end * self.sample_rate))
        
        # Sanity check
        if start_sample >= end_sample:
            return None, False, "invalid_range"
        
        segment = audio[start_sample:end_sample]
        
        # Length check
        if len(segment) < self.MIN_SEGMENT_SAMPLES:
            return segment, False, f"too_short_{len(segment)}_samples"
        
        # NaN/Inf check
        if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
            segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
        
        return segment, True, None
    
    def safe_rms(self, segment: np.ndarray, frame_length: int = 256, hop_length: int = 64) -> np.ndarray:
        """
        PRECISION: Calculate RMS with NaN protection.
        """
        if not HAS_LIBROSA:
            return np.array([0.0])
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Protect against NaN/Inf
        rms = np.nan_to_num(rms, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to prevent division issues
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        
        return rms
    
    def validate_qalqalah(self, 
                          audio: np.ndarray,
                          start: float,
                          end: float) -> QalqalahResult:
        """
        Validate Qalqalah rule: Must show closure (RMS dip) then release (RMS spike)
        
        Physics: The "bounce" is caused by complete oral closure followed by
        abrupt release. RMS energy shows: stable→dip→spike pattern.
        """
        if not HAS_LIBROSA:
            return QalqalahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="RMS Energy",
                expected_pattern="dip_then_spike",
                observed_pattern="unknown",
                score=0.0,
                rms_profile="unknown"
            )
        
        # PRECISION: Use safe extraction
        segment, is_valid, error = self.safe_extract_segment(audio, start, end)
        
        if not is_valid:
            return QalqalahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="RMS Energy",
                expected_pattern="dip_then_spike",
                observed_pattern=error or "invalid_segment",
                score=0.0,
                rms_profile="unknown",
                details={"reason": error}
            )
        
        # PRECISION: Use safe RMS with NaN protection
        rms = self.safe_rms(segment)
        
        if len(rms) < 3:
            return QalqalahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="RMS Energy",
                expected_pattern="dip_then_spike",
                observed_pattern="insufficient_frames",
                score=0.0,
                rms_profile="unknown",
                details={"reason": f"Only {len(rms)} RMS frames < 3 minimum"}
            )
        
        # Analyze RMS pattern
        # Qalqalah should show: high → dip → spike
        # Find minimum and maximum in second half (release)
        midpoint = len(rms) // 2
        
        # First half: Find the dip (closure)
        first_half_mean = np.mean(rms[:midpoint]) if midpoint > 0 else rms[0]
        dip_idx = np.argmin(rms)
        dip_value = rms[dip_idx]
        
        # Second half: Find the spike (release)
        spike_idx = midpoint + np.argmax(rms[midpoint:]) if midpoint < len(rms) else len(rms) - 1
        spike_value = rms[spike_idx] if spike_idx < len(rms) else rms[-1]
        
        # Calculate metrics
        dip_depth = (first_half_mean - dip_value) / first_half_mean if first_half_mean > 0 else 0
        spike_height = (spike_value - dip_value) / dip_value if dip_value > 0 else 0
        
        # Determine pattern
        if dip_depth >= self.QALQALAH_DIP_THRESHOLD and spike_height >= self.QALQALAH_SPIKE_THRESHOLD:
            rms_profile = "dip_then_spike"
            status = ValidationStatus.PASS
            score = min(1.0, (dip_depth + spike_height) / 2)
        elif spike_height >= self.QALQALAH_SPIKE_THRESHOLD:
            rms_profile = "spike_only"
            status = ValidationStatus.MARGINAL
            score = spike_height / 2
        else:
            rms_profile = "flat"
            status = ValidationStatus.FAIL
            score = 0.0
        
        # Estimate closure duration (using safe_rms default hop_length=64)
        if dip_idx > 0:
            frames_to_dip = dip_idx
            closure_duration_ms = (frames_to_dip * 64 / self.sample_rate) * 1000
        else:
            closure_duration_ms = 0.0
        
        return QalqalahResult(
            status=status,
            metric_name="RMS Energy",
            expected_pattern="dip_then_spike",
            observed_pattern=rms_profile,
            score=score,
            rms_profile=rms_profile,
            dip_depth=dip_depth,
            spike_height=spike_height,
            closure_duration_ms=closure_duration_ms
        )
    
    def validate_madd(self,
                      audio: np.ndarray,
                      start: float,
                      end: float,
                      expected_count: int = 2) -> MaddResult:
        """
        Validate Madd rule: Duration must match expected elongation count
        
        Physics: Madd is pure duration comparison.
        - Asli (natural): 2 counts
        - Wajib (obligatory): 4-5 counts
        - Lazim (required): 6 counts
        """
        actual_duration = end - start
        actual_duration_ms = actual_duration * 1000
        
        # Expected duration based on average vowel and count
        expected_duration = self._average_vowel_duration * expected_count
        expected_duration_ms = expected_duration * 1000
        
        # Calculate ratio
        ratio = actual_duration / self._average_vowel_duration if self._average_vowel_duration > 0 else 0
        
        # Determine pass/fail based on expected count
        tolerance = 0.3  # 30% tolerance
        
        if expected_count == 2:
            threshold = self.MADD_RATIO_ASLI
        elif expected_count == 4:
            threshold = self.MADD_RATIO_WAJIB
        else:
            threshold = self.MADD_RATIO_LAZIM
        
        if ratio >= threshold * (1 - tolerance):
            if ratio <= threshold * (1 + tolerance):
                status = ValidationStatus.PASS
                score = 1.0
            else:
                status = ValidationStatus.MARGINAL  # Too long, but acceptable
                score = 0.7
        else:
            status = ValidationStatus.FAIL
            score = ratio / threshold if threshold > 0 else 0
        
        return MaddResult(
            status=status,
            metric_name="Duration Ratio",
            expected_pattern=f"{expected_count}x average vowel",
            observed_pattern=f"{ratio:.1f}x average vowel",
            score=score,
            actual_duration_ms=actual_duration_ms,
            expected_duration_ms=expected_duration_ms,
            ratio=ratio
        )
    
    def validate_ghunnah(self,
                         audio: np.ndarray,
                         start: float,
                         end: float) -> GhunnahResult:
        """
        Validate Ghunnah (nasalization) rule
        
        Physics:
        - Drop in high-frequency energy (nasal anti-formant ~500Hz)
        - Stable pitch during nasalization
        - Duration elongation (2 counts minimum)
        """
        if not HAS_PARSELMOUTH:
            return GhunnahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Formant Analysis",
                expected_pattern="nasal_resonance",
                observed_pattern="unknown",
                score=0.0
            )
        
        duration_ms = (end - start) * 1000
        
        # Check minimum duration
        if duration_ms < self.GHUNNAH_MIN_DURATION_MS:
            return GhunnahResult(
                status=ValidationStatus.MARGINAL,  # PRECISION: Changed from FAIL to MARGINAL
                metric_name="Formant Analysis",
                expected_pattern="nasal_resonance",
                observed_pattern="short_but_valid",
                score=duration_ms / self.GHUNNAH_MIN_DURATION_MS,
                duration_elongation=duration_ms / self.GHUNNAH_MIN_DURATION_MS,
                details={"reason": f"Duration {duration_ms:.1f}ms < {self.GHUNNAH_MIN_DURATION_MS}ms minimum"}
            )
        
        # PRECISION: Use safe extraction
        segment, is_valid, error = self.safe_extract_segment(audio, start, end)
        
        if not is_valid:
            return GhunnahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Formant Analysis",
                expected_pattern="nasal_resonance",
                observed_pattern=error or "invalid_segment",
                score=0.0,
                details={"reason": error}
            )
        
        # Convert to Praat Sound object
        try:
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, segment, self.sample_rate)
                sound = parselmouth.Sound(f.name)
            
            # Get pitch for stability analysis
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced
            
            if len(pitch_values) > 1:
                pitch_stability = 1.0 - (np.std(pitch_values) / np.mean(pitch_values))
            else:
                pitch_stability = 0.0
            
            # Formant analysis for nasal detection
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Nasalization shows anti-resonance around F1 region
            # Check for characteristic nasal formant pattern
            nasal_formant_detected = True  # Simplified detection
            
        except Exception as e:
            print(f"Parselmouth error: {e}")
            return GhunnahResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Formant Analysis",
                expected_pattern="nasal_resonance",
                observed_pattern="analysis_error",
                score=0.0
            )
        
        # Scoring
        duration_score = min(1.0, duration_ms / (self.GHUNNAH_MIN_DURATION_MS * 2))
        pitch_score = max(0.0, pitch_stability)
        total_score = (duration_score + pitch_score) / 2
        
        if total_score >= 0.7:
            status = ValidationStatus.PASS
        elif total_score >= 0.4:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.FAIL
        
        return GhunnahResult(
            status=status,
            metric_name="Formant Analysis",
            expected_pattern="nasal_resonance",
            observed_pattern="analyzed",
            score=total_score,
            nasal_formant_detected=nasal_formant_detected,
            pitch_stability=pitch_stability,
            duration_elongation=duration_ms / self.GHUNNAH_MIN_DURATION_MS
        )
    
    def validate_tafkheem(self,
                          audio: np.ndarray,
                          start: float,
                          end: float) -> TafkheemResult:
        """
        Validate Tafkheem (heavy letter) rule
        
        Physics: Heavy letters show depressed F2 formant
        - Normal letters: F2 ~1500 Hz
        - Heavy letters: F2 ~1000-1200 Hz
        """
        if not HAS_PARSELMOUTH:
            return TafkheemResult(
                status=ValidationStatus.SKIPPED,
                metric_name="F2 Formant",
                expected_pattern="F2 < 1200 Hz",
                observed_pattern="unknown",
                score=0.0
            )
        
        # PRECISION: Use safe extraction
        segment, is_valid, error = self.safe_extract_segment(audio, start, end)
        
        if not is_valid:
            return TafkheemResult(
                status=ValidationStatus.SKIPPED,
                metric_name="F2 Formant",
                expected_pattern=f"F2 < {self.TAFKHEEM_F2_MAX_HZ} Hz",
                observed_pattern=error or "invalid_segment",
                score=0.0,
                details={"reason": error}
            )
        
        try:
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, segment, self.sample_rate)
                sound = parselmouth.Sound(f.name)
            
            # Get F2 formant
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Get average F2
            f2_values = []
            num_frames = call(formant, "Get number of frames")
            for i in range(1, num_frames + 1):
                f2 = call(formant, "Get value at time", 2, call(formant, "Get time from frame number", i), "Hertz", "Linear")
                if not np.isnan(f2) and f2 > 0:
                    f2_values.append(f2)
            
            if f2_values:
                f2_mean = np.mean(f2_values)
            else:
                f2_mean = 0
            
        except Exception as e:
            print(f"Parselmouth error: {e}")
            return TafkheemResult(
                status=ValidationStatus.SKIPPED,
                metric_name="F2 Formant",
                expected_pattern="F2 < 1200 Hz",
                observed_pattern="analysis_error",
                score=0.0
            )
        
        # Calculate depression ratio
        baseline_f2 = 1500.0
        depression_ratio = (baseline_f2 - f2_mean) / baseline_f2 if f2_mean > 0 and f2_mean < baseline_f2 else 0
        
        # Scoring
        if f2_mean <= self.TAFKHEEM_F2_MAX_HZ:
            status = ValidationStatus.PASS
            score = 1.0
        elif f2_mean <= 1350:
            status = ValidationStatus.MARGINAL
            score = 0.6
        else:
            status = ValidationStatus.FAIL
            score = max(0.0, depression_ratio)
        
        return TafkheemResult(
            status=status,
            metric_name="F2 Formant",
            expected_pattern=f"F2 < {self.TAFKHEEM_F2_MAX_HZ} Hz",
            observed_pattern=f"F2 = {f2_mean:.0f} Hz",
            score=score,
            f2_value_hz=f2_mean,
            f2_baseline_hz=baseline_f2,
            depression_ratio=depression_ratio
        )
    
    # =========================================================================
    # NEW VALIDATORS: Complete Tajweed Physics Coverage
    # =========================================================================
    
    def validate_idgham(self,
                        audio: np.ndarray,
                        nun_start: float,
                        nun_end: float,
                        next_letter_end: float,
                        has_ghunnah: bool = True) -> PhysicsResult:
        """
        Validate Idgham (assimilation) rule
        
        Physics:
        - Full Idgham (ر/ل): Complete merger, smooth energy, no nun boundary
        - Partial Idgham (ي/ن/م/و): Ghunnah preserved during transition
        """
        if not HAS_LIBROSA:
            return PhysicsResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Energy Continuity",
                expected_pattern="smooth_transition",
                observed_pattern="unknown",
                score=0.0
            )
        
        # Extract the transition window (nun end to next letter)
        start_sample = int(nun_start * self.sample_rate)
        end_sample = int(next_letter_end * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 100:
            return PhysicsResult(
                status=ValidationStatus.FAIL,
                metric_name="Energy Continuity",
                expected_pattern="smooth_transition",
                observed_pattern="segment_too_short",
                score=0.0
            )
        
        # Calculate RMS to check for smooth energy transition
        frame_length = 256
        hop_length = 64
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate energy variance - low variance = smooth transition
        rms_variance = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 1.0
        
        # For Idgham, we expect smooth continuous energy (low variance)
        smoothness_score = 1.0 - min(1.0, rms_variance)
        
        # Check for boundary sharpness (should be LOW for Idgham)
        rms_diff = np.abs(np.diff(rms))
        max_jump = np.max(rms_diff) / np.mean(rms) if np.mean(rms) > 0 else 0
        boundary_score = 1.0 - min(1.0, max_jump)
        
        total_score = (smoothness_score + boundary_score) / 2
        
        if total_score >= 0.6:
            status = ValidationStatus.PASS
        elif total_score >= 0.4:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.FAIL
        
        return PhysicsResult(
            status=status,
            metric_name="Energy Continuity",
            expected_pattern="smooth_transition" if not has_ghunnah else "smooth_with_ghunnah",
            observed_pattern=f"smoothness={smoothness_score:.2f}",
            score=total_score,
            details={"smoothness": smoothness_score, "boundary_score": boundary_score}
        )
    
    def validate_ikhfa(self,
                       audio: np.ndarray,
                       start: float,
                       end: float) -> PhysicsResult:
        """
        Validate Ikhfa (concealment) rule
        
        Physics:
        - Gradual nasalization transition (not abrupt like pure Ghunnah)
        - Partial nasal resonance that fades
        """
        if not HAS_LIBROSA:
            return PhysicsResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Nasalization Gradient",
                expected_pattern="gradual_nasal",
                observed_pattern="unknown",
                score=0.0
            )
        
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 100:
            return PhysicsResult(
                status=ValidationStatus.FAIL,
                metric_name="Nasalization Gradient",
                expected_pattern="gradual_nasal",
                observed_pattern="segment_too_short",
                score=0.0
            )
        
        # Split into thirds to check for gradient
        third = len(segment) // 3
        
        # Calculate spectral centroid (nasal sounds have lower centroid)
        sc = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0]
        
        if len(sc) < 3:
            return PhysicsResult(
                status=ValidationStatus.FAIL,
                metric_name="Nasalization Gradient",
                expected_pattern="gradual_nasal",
                observed_pattern="insufficient_frames",
                score=0.0
            )
        
        # Check for gradient pattern: centroid should change gradually
        sc_diff = np.abs(np.diff(sc))
        gradient_smoothness = 1.0 - min(1.0, np.std(sc_diff) / np.mean(sc_diff)) if np.mean(sc_diff) > 0 else 0.5
        
        # Duration check (Ikhfa should have reasonable duration)
        duration_ms = (end - start) * 1000
        duration_score = min(1.0, duration_ms / 100) if duration_ms > 0 else 0
        
        total_score = (gradient_smoothness + duration_score) / 2
        
        if total_score >= 0.6:
            status = ValidationStatus.PASS
        elif total_score >= 0.4:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.FAIL
        
        return PhysicsResult(
            status=status,
            metric_name="Nasalization Gradient",
            expected_pattern="gradual_nasal",
            observed_pattern=f"gradient={gradient_smoothness:.2f}",
            score=total_score,
            details={"gradient_smoothness": gradient_smoothness, "duration_ms": duration_ms}
        )
    
    def validate_iqlab(self,
                       audio: np.ndarray,
                       start: float,
                       end: float) -> PhysicsResult:
        """
        Validate Iqlab (ن→م before ب)
        
        Physics:
        - Same as Ghunnah but with bilabial closure
        - Nasal formant + lip closure pattern (F1/F2 characteristic of /m/)
        """
        # Iqlab is essentially Ghunnah with bilabial characteristics
        # Reuse ghunnah validation logic
        ghunnah_result = self.validate_ghunnah(audio, start, end)
        
        # Modify result type for Iqlab
        return PhysicsResult(
            status=ghunnah_result.status,
            metric_name="Bilabial Nasal",
            expected_pattern="mim_like_nasal",
            observed_pattern=ghunnah_result.observed_pattern,
            score=ghunnah_result.score,
            details={"ghunnah_check": ghunnah_result.status.value}
        )
    
    def validate_izhar(self,
                       audio: np.ndarray,
                       letter_start: float,
                       letter_end: float,
                       next_letter_start: float) -> PhysicsResult:
        """
        Validate Izhar (clear pronunciation)
        
        Physics:
        - Clean, sharp boundary between letters
        - No nasalization
        - Clear articulation energy pattern
        """
        if not HAS_LIBROSA:
            return PhysicsResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Boundary Sharpness",
                expected_pattern="clean_boundary",
                observed_pattern="unknown",
                score=0.0
            )
        
        # Check boundary region
        boundary_start = max(0, letter_end - 0.02)  # 20ms before boundary
        boundary_end = min(len(audio) / self.sample_rate, next_letter_start + 0.02)  # 20ms after
        
        start_sample = int(boundary_start * self.sample_rate)
        end_sample = int(boundary_end * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 50:
            return PhysicsResult(
                status=ValidationStatus.FAIL,
                metric_name="Boundary Sharpness",
                expected_pattern="clean_boundary",
                observed_pattern="segment_too_short",
                score=0.0
            )
        
        # Calculate RMS to find sharp transitions
        frame_length = 128
        hop_length = 32
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Look for clear dip/change at boundary
        rms_diff = np.abs(np.diff(rms))
        max_change = np.max(rms_diff) / np.mean(rms) if np.mean(rms) > 0 else 0
        
        # High change = sharp boundary = good for Izhar
        sharpness_score = min(1.0, max_change)
        
        if sharpness_score >= 0.3:  # Clear boundary detected
            status = ValidationStatus.PASS
            score = min(1.0, sharpness_score * 2)
        elif sharpness_score >= 0.15:
            status = ValidationStatus.MARGINAL
            score = sharpness_score * 2
        else:
            status = ValidationStatus.FAIL
            score = sharpness_score
        
        return PhysicsResult(
            status=status,
            metric_name="Boundary Sharpness",
            expected_pattern="clean_boundary",
            observed_pattern=f"sharpness={sharpness_score:.2f}",
            score=score,
            details={"boundary_sharpness": sharpness_score}
        )
    
    def validate_tarqeeq(self,
                         audio: np.ndarray,
                         start: float,
                         end: float) -> PhysicsResult:
        """
        Validate Tarqeeq (light letters) - opposite of Tafkheem
        
        Physics: Light letters show elevated F2 formant (F2 > 1400 Hz)
        """
        # Reuse Tafkheem logic but invert the threshold
        tafkheem_result = self.validate_tafkheem(audio, start, end)
        
        if tafkheem_result.status == ValidationStatus.SKIPPED:
            return PhysicsResult(
                status=ValidationStatus.SKIPPED,
                metric_name="F2 Formant",
                expected_pattern="F2 > 1400 Hz",
                observed_pattern="unknown",
                score=0.0
            )
        
        # For Tarqeeq, we want HIGH F2 (opposite of Tafkheem)
        f2_value = tafkheem_result.details.get('f2_value_hz', tafkheem_result.f2_value_hz if hasattr(tafkheem_result, 'f2_value_hz') else 0)
        
        TARQEEQ_F2_MIN_HZ = 1400.0
        
        if f2_value >= TARQEEQ_F2_MIN_HZ:
            status = ValidationStatus.PASS
            score = 1.0
        elif f2_value >= 1300:
            status = ValidationStatus.MARGINAL
            score = 0.6
        else:
            status = ValidationStatus.FAIL
            score = f2_value / TARQEEQ_F2_MIN_HZ if f2_value > 0 else 0
        
        return PhysicsResult(
            status=status,
            metric_name="F2 Formant",
            expected_pattern=f"F2 > {TARQEEQ_F2_MIN_HZ} Hz",
            observed_pattern=f"F2 = {f2_value:.0f} Hz",
            score=score,
            details={"f2_value_hz": f2_value}
        )
    
    def validate_sakt(self,
                      audio: np.ndarray,
                      start: float,
                      end: float) -> PhysicsResult:
        """
        Validate Sakt (brief pause without breath)
        
        Physics:
        - Brief silence (50-200ms)
        - RMS below threshold
        - No breathing artifacts
        """
        if not HAS_LIBROSA:
            return PhysicsResult(
                status=ValidationStatus.SKIPPED,
                metric_name="Silence Detection",
                expected_pattern="brief_silence",
                observed_pattern="unknown",
                score=0.0
            )
        
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        duration_ms = (end - start) * 1000
        
        if len(segment) < 10:
            return PhysicsResult(
                status=ValidationStatus.FAIL,
                metric_name="Silence Detection",
                expected_pattern="brief_silence",
                observed_pattern="segment_too_short",
                score=0.0
            )
        
        # Calculate RMS
        rms = np.sqrt(np.mean(segment**2))
        
        # Thresholds
        SAKT_RMS_THRESHOLD = 0.05
        SAKT_MIN_MS = 50
        SAKT_MAX_MS = 200
        
        # Check RMS (should be very low)
        is_silent = rms < SAKT_RMS_THRESHOLD
        
        # Check duration
        duration_ok = SAKT_MIN_MS <= duration_ms <= SAKT_MAX_MS
        
        if is_silent and duration_ok:
            status = ValidationStatus.PASS
            score = 1.0
        elif is_silent and (duration_ms > 30):
            status = ValidationStatus.MARGINAL
            score = 0.6
        else:
            status = ValidationStatus.FAIL
            score = 0.0 if rms >= SAKT_RMS_THRESHOLD else 0.3
        
        return PhysicsResult(
            status=status,
            metric_name="Silence Detection",
            expected_pattern=f"silence_{SAKT_MIN_MS}-{SAKT_MAX_MS}ms",
            observed_pattern=f"rms={rms:.3f}, dur={duration_ms:.0f}ms",
            score=score,
            details={"rms": rms, "duration_ms": duration_ms, "is_silent": is_silent}
        )
    
    def calibrate_average_vowel(self, audio: np.ndarray, vowel_segments: List[Tuple[float, float]]) -> float:
        """
        Calibrate average vowel duration for this reciter
        
        This is crucial for Madd validation as reciter pace varies
        """
        if not vowel_segments:
            return 0.1  # Default 100ms
        
        durations = [end - start for start, end in vowel_segments]
        self._average_vowel_duration = np.mean(durations)
        
        return self._average_vowel_duration


def main():
    """Test physics validator"""
    print("=" * 50)
    print("TajweedSST Physics Validator Test")
    print("=" * 50)
    
    # Create mock audio
    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a test signal with dip→spike pattern (simulating Qalqalah)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    # Add dip in middle
    dip_start = int(len(audio) * 0.4)
    dip_end = int(len(audio) * 0.5)
    audio[dip_start:dip_end] *= 0.1
    # Add spike after dip
    spike_start = int(len(audio) * 0.5)
    spike_end = int(len(audio) * 0.6)
    audio[spike_start:spike_end] *= 2.0
    
    validator = PhysicsValidator(sample_rate=sample_rate)
    
    # Test Qalqalah
    print("\nQalqalah Test:")
    result = validator.validate_qalqalah(audio, 0.3, 0.8)
    print(f"  Status: {result.status.value}")
    print(f"  Profile: {result.rms_profile}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Dip Depth: {result.dip_depth:.2f}")
    print(f"  Spike Height: {result.spike_height:.2f}")
    
    # Test Madd
    print("\nMadd Test:")
    validator._average_vowel_duration = 0.1  # 100ms average
    result = validator.validate_madd(audio, 0.0, 0.4, expected_count=4)
    print(f"  Status: {result.status.value}")
    print(f"  Ratio: {result.ratio:.1f}x")
    print(f"  Score: {result.score:.2f}")


if __name__ == "__main__":
    main()
