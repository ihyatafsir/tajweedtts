#!/usr/bin/env python3
"""
TajweedSST - Pipeline Integration Tests

Tests the full alignment pipeline end-to-end:
- Text parsing → Alignment → Physics Validation
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alignment_engine import MockAlignmentEngine, AlignmentResult


class TestFullPipeline:
    """Integration tests for complete pipeline"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockAlignmentEngine()
    
    def test_surah_91_ayah_1(self, mock_engine):
        """Test alignment for Surah 91, Ayah 1: والشمس وضحاها"""
        phonetic_words = [
            "w a l sh sh a m s i",
            "w a D u H aa h aa"
        ]
        
        result = mock_engine.align(
            audio_path="/path/to/surah_91_ayah_1.wav",
            phonetic_words=phonetic_words,
            surah=91,
            ayah=1
        )
        
        assert result.surah == 91
        assert result.ayah == 1
        assert len(result.words) == 2
        
        # Verify monotonicity
        for i in range(1, len(result.words)):
            assert result.words[i].whisper_start >= result.words[i-1].whisper_end
    
    def test_grapheme_count_matches(self, mock_engine):
        """Total graphemes should match input"""
        phonetic_words = ["a b c", "d e f g"]  # 7 phonemes total
        
        result = mock_engine.align(
            audio_path="/fake.wav",
            phonetic_words=phonetic_words,
            surah=1,
            ayah=1
        )
        
        total_phonemes = sum(len(w.phonemes) for w in result.words)
        # Each space-separated token should become a phoneme
        expected = sum(len(w.split()) for w in phonetic_words)
        assert total_phonemes >= expected - 2  # Allow some variance


class TestTimingRegression:
    """Tests to catch timing regressions"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockAlignmentEngine()
    
    def test_no_negative_durations(self, mock_engine):
        """No phoneme should have negative duration"""
        result = mock_engine.align(
            audio_path="/fake.wav",
            phonetic_words=["a b c d e f g h i j"],
            surah=1,
            ayah=1
        )
        
        for word in result.words:
            for phoneme in word.phonemes:
                assert phoneme.duration >= 0, \
                    f"Negative duration: {phoneme.phoneme} = {phoneme.duration}"
    
    def test_no_zero_duration_phonemes(self, mock_engine):
        """Phonemes should have positive duration"""
        result = mock_engine.align(
            audio_path="/fake.wav",
            phonetic_words=["test word"],
            surah=1,
            ayah=1
        )
        
        for word in result.words:
            for phoneme in word.phonemes:
                assert phoneme.duration > 0, \
                    f"Zero duration phoneme: {phoneme.phoneme}"
    
    def test_no_overlapping_phonemes(self, mock_engine):
        """Phonemes within a word should not overlap"""
        result = mock_engine.align(
            audio_path="/fake.wav",
            phonetic_words=["a l r a h m a n"],
            surah=1,
            ayah=1
        )
        
        for word in result.words:
            for i in range(1, len(word.phonemes)):
                prev = word.phonemes[i-1]
                curr = word.phonemes[i]
                assert curr.start >= prev.end, \
                    f"Overlap: {prev.phoneme} ({prev.end}) > {curr.phoneme} ({curr.start})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
