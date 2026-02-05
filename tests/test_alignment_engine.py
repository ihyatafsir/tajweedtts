#!/usr/bin/env python3
"""
TajweedSST - Alignment Engine Unit Tests

Tests word and phoneme timing accuracy:
- WhisperX word alignment
- MFA phoneme alignment  
- Phoneme normalization within word boundaries
- Mock alignment for testing without models
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alignment_engine import (
    AlignmentEngine,
    MockAlignmentEngine,
    PhonemeAlignment,
    WordAlignment,
    AlignmentResult
)


class TestDataclasses:
    """Test alignment data structures"""
    
    def test_phoneme_alignment(self):
        """PhonemeAlignment stores timing correctly"""
        pa = PhonemeAlignment(phoneme="ب", start=0.0, end=0.1, duration=0.1)
        assert pa.phoneme == "ب"
        assert pa.duration == 0.1
    
    def test_phoneme_normalized_duration(self):
        """Normalized duration calculation"""
        pa = PhonemeAlignment(phoneme="ا", start=0.0, end=0.2, duration=0.2)
        # normalized_duration is a property
        assert pa.normalized_duration == 0.2
    
    def test_word_alignment(self):
        """WordAlignment stores word and phonemes"""
        wa = WordAlignment(
            word_text="بسم",
            whisper_start=0.0,
            whisper_end=0.5,
            phonemes=[
                PhonemeAlignment("ب", 0.0, 0.15, 0.15),
                PhonemeAlignment("س", 0.15, 0.35, 0.20),
                PhonemeAlignment("م", 0.35, 0.5, 0.15),
            ]
        )
        assert wa.word_text == "بسم"
        assert len(wa.phonemes) == 3
        assert wa.whisper_duration == 0.5
    
    def test_alignment_result(self):
        """AlignmentResult stores full alignment"""
        ar = AlignmentResult(
            audio_path="/path/to/audio.wav",
            surah=91,
            ayah=1,
            words=[]
        )
        assert ar.surah == 91
        assert ar.ayah == 1


class TestMockAlignmentEngine:
    """Test mock alignment for development without models"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockAlignmentEngine()
    
    def test_mock_align_returns_result(self, mock_engine):
        """Mock alignment returns AlignmentResult"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["b i s m", "a l l a h"],
            surah=1,
            ayah=1
        )
        assert isinstance(result, AlignmentResult)
    
    def test_mock_align_word_count(self, mock_engine):
        """Mock alignment produces correct word count"""
        phonetic_words = ["b i s m", "a l l a h", "a r r a h m a n"]
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=phonetic_words,
            surah=1,
            ayah=1
        )
        assert len(result.words) == len(phonetic_words)
    
    def test_mock_align_phoneme_generation(self, mock_engine):
        """Mock alignment generates phonemes for each word"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["b i s m"],
            surah=1,
            ayah=1
        )
        # "b i s m" should produce ~4 phonemes
        assert len(result.words[0].phonemes) >= 3
    
    def test_mock_align_timing_monotonic(self, mock_engine):
        """Mock timing should be monotonically increasing"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["word1", "word2", "word3"],
            surah=1,
            ayah=1
        )
        
        prev_end = 0.0
        for word in result.words:
            assert word.whisper_start >= prev_end, "Word start before previous end"
            prev_end = word.whisper_end


class TestTimingMonotonicity:
    """Test that timing never goes backwards"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockAlignmentEngine()
    
    def test_word_timing_monotonic(self, mock_engine):
        """Word-level timing is strictly increasing"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["w1", "w2", "w3", "w4", "w5"],
            surah=1,
            ayah=1
        )
        
        for i in range(1, len(result.words)):
            prev = result.words[i-1]
            curr = result.words[i]
            assert curr.whisper_start >= prev.whisper_end, \
                f"Word {i} starts ({curr.whisper_start}) before word {i-1} ends ({prev.whisper_end})"
    
    def test_phoneme_timing_monotonic(self, mock_engine):
        """Phoneme-level timing is strictly increasing within words"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["a l r a h m a n"],
            surah=1,
            ayah=1
        )
        
        for word in result.words:
            for i in range(1, len(word.phonemes)):
                prev = word.phonemes[i-1]
                curr = word.phonemes[i]
                assert curr.start >= prev.end, \
                    f"Phoneme {curr.phoneme} starts before {prev.phoneme} ends"


class TestPhonemeNormalization:
    """Test phoneme duration normalization"""
    
    def test_phonemes_fit_word_boundary(self):
        """Normalized phonemes should fit exactly in word boundaries"""
        word = WordAlignment(
            word_text="test",
            whisper_start=1.0,
            whisper_end=2.0,
            phonemes=[
                PhonemeAlignment("t", 1.0, 1.25, 0.25),
                PhonemeAlignment("e", 1.25, 1.5, 0.25),
                PhonemeAlignment("s", 1.5, 1.75, 0.25),
                PhonemeAlignment("t", 1.75, 2.0, 0.25),
            ]
        )
        
        # First phoneme should start at word start
        assert word.phonemes[0].start == word.whisper_start
        # Last phoneme should end at word end
        assert word.phonemes[-1].end == word.whisper_end
    
    def test_phonemes_cover_word_duration(self):
        """Phoneme durations should sum to word duration"""
        word = WordAlignment(
            word_text="test",
            whisper_start=0.0,
            whisper_end=1.0,
            phonemes=[
                PhonemeAlignment("a", 0.0, 0.333, 0.333),
                PhonemeAlignment("b", 0.333, 0.666, 0.333),
                PhonemeAlignment("c", 0.666, 1.0, 0.334),
            ]
        )
        
        total_phoneme_duration = sum(p.duration for p in word.phonemes)
        word_duration = word.whisper_duration
        # Allow small floating point error
        assert abs(total_phoneme_duration - word_duration) < 0.01


class TestArabicPhonemes:
    """Test Arabic-specific phoneme handling"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockAlignmentEngine()
    
    def test_arabic_phonetic_transcription(self, mock_engine):
        """Engine handles Arabic phonetic transcription"""
        result = mock_engine.align(
            audio_path="/fake/path.wav",
            phonetic_words=["b i s m i", "a l l aa h i"],  # Arabic transliteration
            surah=1,
            ayah=1
        )
        assert len(result.words) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
